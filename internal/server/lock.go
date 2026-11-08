package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/hofftodd/ai-swap/internal/cluster"
)

// LockHeader carries the token of the client holding this machine's
// reservation; requests with it are served while the machine is reserved.
const LockHeader = "X-AI-Swap-Lock"

// A reservation hands this machine to one client, such as a benchmark run:
// while it lasts, only requests carrying its token are served or may load
// and unload models here. Everyone else gets 503 with Retry-After, saying who
// holds the machine and why. Idle unloads and resident reloads pause. The
// reservation lapses if not renewed (or used) within its TTL, so a crashed
// client can't hold the machine forever; releasing it brings back the
// resident models it unloaded.
type reservation struct {
	mu       sync.Mutex
	token    string
	info     cluster.LockInfo
	ttl      time.Duration
	inflight int
}

func newToken() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// current returns the active reservation, expiring it first if it lapsed.
func (s *Server) currentLock() *cluster.LockInfo {
	l := &s.lock
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.token == "" {
		return nil
	}
	if l.inflight == 0 && time.Now().After(l.info.Expires) {
		s.log.Warn("reservation lapsed", "owner", l.info.Owner)
		s.releaseLocked()
		return nil
	}
	info := l.info
	return &info
}

// releaseLocked drops the reservation; the caller holds s.lock.mu.
func (s *Server) releaseLocked() {
	s.lock.token, s.lock.info, s.lock.inflight = "", cluster.LockInfo{}, 0
	s.mgr.SetPaused(false)
}

// lockGate reports whether r may use this machine now. A request carrying the
// token renews the reservation and is tracked until done is called.
func (s *Server) lockGate(r *http.Request) (ok bool, done func(), held *cluster.LockInfo) {
	held = s.currentLock()
	if held == nil {
		return true, func() {}, nil
	}
	l := &s.lock
	l.mu.Lock()
	defer l.mu.Unlock()
	if r.Header.Get(LockHeader) != l.token {
		return false, nil, held
	}
	l.inflight++
	l.info.Expires = time.Now().Add(l.ttl)
	return true, func() {
		l.mu.Lock()
		l.inflight--
		if l.token != "" {
			l.info.Expires = time.Now().Add(l.ttl)
		}
		l.mu.Unlock()
	}, nil
}

// reservedError answers a request refused because of someone else's reservation.
func (s *Server) reservedError(w http.ResponseWriter, r *http.Request, held *cluster.LockInfo) {
	w.Header().Set("Retry-After", "60")
	msg := fmt.Sprintf("%s is reserved by %s", s.cfg.Node.Name, held.Owner)
	if held.Reason != "" {
		msg += " (" + held.Reason + ")"
	}
	msg += fmt.Sprintf(" since %s; try again later", held.Since.Format("15:04"))
	s.apiError(w, r, http.StatusServiceUnavailable, "api_error", msg)
}

// POST /api/lock: take the reservation, or renew it with the token header.
func (s *Server) handleLock(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Owner      string   `json:"owner"`
		Reason     string   `json:"reason"`
		Models     []string `json:"models"`
		TTLSeconds int      `json:"ttl_seconds"`
	}
	if r.ContentLength != 0 {
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "body must be JSON: " + err.Error()})
			return
		}
	}
	held := s.currentLock()
	l := &s.lock
	l.mu.Lock()
	defer l.mu.Unlock()
	token := r.Header.Get(LockHeader)
	switch {
	case held != nil && token == l.token: // renew
		if body.TTLSeconds > 0 {
			l.ttl = time.Duration(body.TTLSeconds) * time.Second
		}
		if body.Reason != "" {
			l.info.Reason = body.Reason
		}
		l.info.Expires = time.Now().Add(l.ttl)
	case held != nil:
		writeJSON(w, http.StatusConflict, map[string]any{
			"error": fmt.Sprintf("%s is already reserved by %s", s.cfg.Node.Name, held.Owner), "lock": held})
		return
	default:
		if body.Owner == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": `"owner" is required`})
			return
		}
		ttl := 5 * time.Minute
		if body.TTLSeconds > 0 {
			ttl = time.Duration(body.TTLSeconds) * time.Second
		}
		now := time.Now()
		l.token, l.ttl, l.inflight = newToken(), ttl, 0
		l.info = cluster.LockInfo{Owner: body.Owner, Reason: body.Reason, Models: body.Models, Since: now, Expires: now.Add(ttl)}
		s.mgr.SetPaused(true)
		s.log.Info("reserved", "owner", body.Owner, "reason", body.Reason, "ttl", ttl)
	}
	writeJSON(w, http.StatusOK, map[string]any{"token": l.token, "lock": l.info})
}

// DELETE /api/lock: release with the token, or ?force=1 (from the UI, for a
// reservation whose owner is gone).
func (s *Server) handleUnlock(w http.ResponseWriter, r *http.Request) {
	// ?node=<peer> releases that machine's reservation (from the UI).
	if n := r.URL.Query().Get("node"); n != "" && n != s.cfg.Node.Name {
		if r.Header.Get(cluster.HopHeader) != "" {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown node " + n})
			return
		}
		for _, p := range s.cluster.Peers() {
			if p.Name == n || (p.Node != nil && p.Node.Name == n) {
				u, err := url.Parse(p.URL)
				if err != nil {
					break
				}
				s.proxy(w, r, nil, u, true)
				s.cluster.Refresh(u)
				return
			}
		}
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown node " + n})
		return
	}
	if s.currentLock() == nil {
		writeJSON(w, http.StatusOK, map[string]any{"lock": nil})
		return
	}
	l := &s.lock
	l.mu.Lock()
	defer l.mu.Unlock()
	if r.Header.Get(LockHeader) != l.token && r.URL.Query().Get("force") == "" {
		writeJSON(w, http.StatusForbidden, map[string]string{"error": "wrong or missing " + LockHeader + " token (add ?force=1 to release anyway)"})
		return
	}
	s.log.Info("released", "owner", l.info.Owner)
	s.releaseLocked()
	writeJSON(w, http.StatusOK, map[string]any{"lock": nil})
}

func (s *Server) handleLockStatus(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"lock": s.currentLock()})
}

// ExpireLocks lets a lapsed reservation go even when no requests arrive.
func (s *Server) ExpireLocks(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.currentLock()
		}
	}
}
