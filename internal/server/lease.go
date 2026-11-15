package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"sort"
	"sync"
	"time"

	"github.com/hofftodd/ai-swap/internal/cluster"
)

// A lease keeps one model loaded for a client across many requests, such as
// an agent step that pauses between turns while its tools run. While a lease
// lasts the model counts as busy: loads that need its memory wait instead of
// evicting it, and the idle reaper leaves it alone. A lease lapses unless
// renewed within its TTL, so a crashed client can't pin a model forever.
//
// A lease does not stop an explicit unload (from the UI or /api/unload): that
// is a person deciding, and wins.
type lease struct {
	id      string
	info    cluster.LeaseInfo
	ttl     time.Duration
	release func()
}

type leases struct {
	mu sync.Mutex
	m  map[string]*lease
}

const (
	defaultLeaseTTL = 10 * time.Minute
	maxLeaseTTL     = time.Hour
)

func leaseTTL(seconds int) time.Duration {
	ttl := defaultLeaseTTL
	if seconds > 0 {
		ttl = time.Duration(seconds) * time.Second
	}
	return min(ttl, maxLeaseTTL)
}

// leaseList returns the active leases (expiring lapsed ones first), oldest first.
func (s *Server) leaseList() []cluster.LeaseInfo {
	s.expireLeases()
	s.leases.mu.Lock()
	defer s.leases.mu.Unlock()
	out := make([]cluster.LeaseInfo, 0, len(s.leases.m))
	for _, l := range s.leases.m {
		out = append(out, l.info)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Since.Before(out[j].Since) })
	return out
}

func (s *Server) expireLeases() {
	now := time.Now()
	s.leases.mu.Lock()
	var lapsed []*lease
	for id, l := range s.leases.m {
		if now.After(l.info.Expires) {
			lapsed = append(lapsed, l)
			delete(s.leases.m, id)
		}
	}
	s.leases.mu.Unlock()
	for _, l := range lapsed {
		s.log.Warn("lease lapsed", "model", l.info.Model, "owner", l.info.Owner)
		l.release()
	}
}

// POST /api/leases: take a lease on a model, or renew one with "lease_id".
// "model@peer" (or a model only a peer has) is forwarded to that peer, which
// holds the lease; the client renews and releases through the same name.
func (s *Server) handleLease(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Model      string `json:"model"`
		Owner      string `json:"owner"`
		Reason     string `json:"reason"`
		TTLSeconds int    `json:"ttl_seconds"`
		LeaseID    string `json:"lease_id"`
	}
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<16))
	if err == nil {
		err = json.Unmarshal(raw, &body)
	}
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "body must be JSON: " + err.Error()})
		return
	}
	if body.LeaseID != "" && body.Model == "" {
		s.renewLease(w, body.LeaseID, body.TTLSeconds)
		return
	}
	if body.Model == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": `"model" is required`})
		return
	}
	rt, ok := s.locate(body.Model, r.Header.Get(cluster.HopHeader) == "")
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown model " + body.Model})
		return
	}
	if rt.peer != nil {
		s.forwardLease(w, r, raw, rt)
		return
	}
	if body.LeaseID != "" {
		s.renewLease(w, body.LeaseID, body.TTLSeconds)
		return
	}
	if body.Owner == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": `"owner" is required`})
		return
	}
	allowed, done, held := s.lockGate(r)
	if !allowed {
		s.reservedError(w, r, held)
		return
	}
	defer done()
	// Acquire loads the model if needed (waiting like any request would) and
	// marks it in flight until the lease ends. It must outlive this request.
	_, release, err := s.mgr.Acquire(r.Context(), rt.local)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	ttl := leaseTTL(body.TTLSeconds)
	now := time.Now()
	l := &lease{
		id:      newToken(),
		ttl:     ttl,
		release: release,
		info: cluster.LeaseInfo{
			Model: rt.local, Owner: body.Owner, Reason: body.Reason,
			Since: now, Expires: now.Add(ttl),
		},
	}
	s.leases.mu.Lock()
	if s.leases.m == nil {
		s.leases.m = map[string]*lease{}
	}
	s.leases.m[l.id] = l
	s.leases.mu.Unlock()
	s.log.Info("leased", "model", rt.local, "owner", body.Owner, "ttl", ttl)
	writeJSON(w, http.StatusOK, map[string]any{"lease_id": l.id, "lease": l.info, "node": s.cfg.Node.Name})
}

func (s *Server) renewLease(w http.ResponseWriter, id string, ttlSeconds int) {
	s.expireLeases()
	s.leases.mu.Lock()
	defer s.leases.mu.Unlock()
	l, ok := s.leases.m[id]
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "no such lease (it may have lapsed)"})
		return
	}
	if ttlSeconds > 0 {
		l.ttl = leaseTTL(ttlSeconds)
	}
	l.info.Expires = time.Now().Add(l.ttl)
	writeJSON(w, http.StatusOK, map[string]any{"lease_id": l.id, "lease": l.info, "node": s.cfg.Node.Name})
}

// forwardLease sends a lease request for a peer's model to that peer, with
// the model renamed to the peer's own name for it.
func (s *Server) forwardLease(w http.ResponseWriter, r *http.Request, raw []byte, rt route) {
	var m map[string]any
	_ = json.Unmarshal(raw, &m)
	m["model"] = rt.remote
	b, _ := json.Marshal(m)
	s.proxy(w, r, b, rt.peer, true)
}

// DELETE /api/leases/{id}: release a lease. ?model= forwards the release to
// the machine holding that model's lease (as the take was forwarded).
func (s *Server) handleLeaseRelease(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if name := r.URL.Query().Get("model"); name != "" {
		if rt, ok := s.locate(name, r.Header.Get(cluster.HopHeader) == ""); ok && rt.peer != nil {
			q := url.Values{"model": {rt.remote}}
			r.URL.RawQuery = q.Encode()
			s.proxy(w, r, nil, rt.peer, true)
			return
		}
	}
	s.leases.mu.Lock()
	l, ok := s.leases.m[id]
	delete(s.leases.m, id)
	s.leases.mu.Unlock()
	if !ok {
		writeJSON(w, http.StatusOK, map[string]any{"released": false})
		return
	}
	l.release()
	s.log.Info("lease released", "model", l.info.Model, "owner", l.info.Owner)
	writeJSON(w, http.StatusOK, map[string]any{"released": true})
}

func (s *Server) handleLeaseStatus(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"leases": s.leaseList()})
}

// ExpireLeases lets lapsed leases go even when no requests arrive.
func (s *Server) ExpireLeases(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.expireLeases()
		}
	}
}

// ReleaseLeases drops every lease (on shutdown).
func (s *Server) ReleaseLeases() {
	s.leases.mu.Lock()
	all := s.leases.m
	s.leases.m = nil
	s.leases.mu.Unlock()
	for _, l := range all {
		l.release()
	}
}
