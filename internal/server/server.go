// Package server is the HTTP front door: OpenAI- and Anthropic-compatible
// inference endpoints that route to local or peer models, a small management
// API, and the embedded chat UI.
package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/hofftodd/ai-swap/internal/cluster"
	"github.com/hofftodd/ai-swap/internal/config"
	"github.com/hofftodd/ai-swap/internal/hostinfo"
	"github.com/hofftodd/ai-swap/internal/scheduler"
	"github.com/hofftodd/ai-swap/internal/stats"
	"github.com/hofftodd/ai-swap/internal/version"
)

const maxBody = 64 << 20

// inferencePaths are forwarded to backends, routed by the body's "model".
var inferencePaths = []string{
	"/v1/chat/completions",
	"/v1/completions",
	"/v1/embeddings",
	"/v1/responses",
	"/v1/messages",
	"/v1/messages/count_tokens",
}

// Server wires the scheduler and cluster to HTTP.
type Server struct {
	cfg       *config.Config
	mgr       *scheduler.Manager
	cluster   *cluster.Cluster
	log       *slog.Logger
	transport http.RoundTripper
	ui        fs.FS
	stats     *stats.Recorder
	lock      reservation // see lock.go
}

// New builds the server. ui may be nil.
func New(cfg *config.Config, mgr *scheduler.Manager, cl *cluster.Cluster, ui fs.FS, log *slog.Logger) *Server {
	return &Server{cfg: cfg, mgr: mgr, cluster: cl, log: log, transport: cluster.NewTransport(), ui: ui, stats: stats.NewRecorder()}
}

// Handler returns the root HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	for _, p := range inferencePaths {
		mux.HandleFunc("POST "+p, s.handleInference)
	}
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, "ok") })
	mux.HandleFunc("GET /api/node", s.handleNode)
	mux.HandleFunc("GET /api/status", s.handleStatus)
	mux.HandleFunc("GET /api/clients/{client}", s.handleClientConfig)
	mux.HandleFunc("POST /api/models/{name}/load", s.handleLoad)
	mux.HandleFunc("POST /api/models/{name}/unload", s.handleUnload)
	mux.HandleFunc("GET /api/models/{name}/logs", s.handleLogs)
	mux.HandleFunc("GET /api/models/{name}/stats", s.handleStats)
	mux.HandleFunc("POST /api/unload", s.handleUnloadAll)
	mux.HandleFunc("GET /api/lock", s.handleLockStatus)
	mux.HandleFunc("POST /api/lock", s.handleLock)
	mux.HandleFunc("DELETE /api/lock", s.handleUnlock)
	// llama-swap compatibility (llamaswap.go).
	mux.HandleFunc("GET /running", s.handleRunning)
	mux.HandleFunc("GET /unload", s.handleUnloadAll)
	mux.HandleFunc("POST /unload", s.handleUnloadAll)
	mux.HandleFunc("GET /upstream/{name}/{path...}", s.handleUpstream)
	if s.ui != nil {
		mux.Handle("GET /", http.FileServerFS(s.ui))
	}
	return s.logRequests(mux)
}

// route describes where a request for a model should go.
type route struct {
	local  string   // local model name, or ""
	peer   *url.URL // peer base URL when not local
	remote string   // model name on the peer
}

// locate finds a model for the management endpoints (load, unload, logs,
// stats): this machine's copy if it has one, else a peer's ("model@node" pins
// the machine). Unlike resolve it never moves to another machine's copy,
// since an unpinned name there means the copy here.
func (s *Server) locate(name string, allowPeers bool) (route, bool) {
	want, node, pinned := strings.Cut(name, "@")
	if !pinned || node == s.cfg.Node.Name {
		if m, ok := s.cfg.Resolve(want); ok {
			if inst, _ := s.mgr.Get(m.Name); inst.Engine.Supported() {
				return route{local: m.Name}, true
			}
		}
	}
	if allowPeers {
		if u, remote, ok := s.cluster.Find(name); ok {
			return route{peer: u, remote: remote}, true
		}
	}
	return route{}, false
}

// resolve finds the model for a client-supplied name ("model", alias, or
// "model@node"). A pinned name goes to that machine. Otherwise, when several
// machines have the model, place picks the one that can serve it without a
// swap (see placement.go); a request carrying this machine's reservation
// token is always served here, where its holder reserved.
func (s *Server) resolve(r *http.Request, name string, allowPeers bool) (route, bool) {
	want, _, pinned := strings.Cut(name, "@")
	if pinned {
		return s.locate(name, allowPeers)
	}
	cands := s.candidates(r, want, allowPeers)
	if s.holdsLock(r) {
		for _, c := range cands {
			if c.local {
				return route{local: c.model}, true
			}
		}
	}
	c, ok := place(cands)
	switch {
	case !ok:
		return route{}, false
	case c.local:
		return route{local: c.model}, true
	default:
		return route{peer: c.peer, remote: c.model}, true
	}
}

func (s *Server) handleInference(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		s.apiError(w, r, http.StatusRequestEntityTooLarge, "invalid_request_error", err.Error())
		return
	}
	var head struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(body, &head); err != nil {
		s.apiError(w, r, http.StatusBadRequest, "invalid_request_error", "body must be JSON: "+err.Error())
		return
	}
	if head.Model == "" {
		s.apiError(w, r, http.StatusBadRequest, "invalid_request_error", `"model" is required`)
		return
	}

	rt, ok := s.resolve(r, head.Model, r.Header.Get(cluster.HopHeader) == "")
	if !ok {
		s.notFound(w, r, head.Model)
		return
	}

	if rt.peer != nil {
		if rt.remote != head.Model {
			body = rewriteModel(body, rt.remote)
		}
		s.proxy(w, r, body, rt.peer, true)
		s.cluster.Refresh(rt.peer)
		return
	}

	allowed, done, held := s.lockGate(r)
	if !allowed {
		s.reservedError(w, r, held)
		return
	}
	defer done()
	start := time.Now()
	port, release, err := s.mgr.Acquire(r.Context(), rt.local)
	if err != nil {
		code := http.StatusServiceUnavailable
		if errors.Is(err, r.Context().Err()) {
			code = 499 // client went away while waiting for the model
		}
		s.apiError(w, r, code, "api_error", err.Error())
		return
	}
	defer release()

	if strings.HasPrefix(r.URL.Path, "/v1/messages") {
		body = foldSystemMessages(body)
	}
	m := s.cfg.Models[rt.local]
	switch {
	case m.UpstreamModel != "":
		body = rewriteModel(body, m.UpstreamModel)
	case head.Model != rt.local:
		body = rewriteModel(body, rt.local)
	}
	// Time to first token is measured from when the model is ready, so a
	// cold load doesn't swamp the figure (load time is reported separately).
	mw := &meter{ResponseWriter: w, start: time.Now(), code: http.StatusOK}
	s.proxy(mw, r, body, &url.URL{Scheme: "http", Host: "127.0.0.1:" + strconv.Itoa(port)}, false)
	if r.URL.Path != "/v1/messages/count_tokens" {
		s.stats.Observe(rt.local, mw.sample(r.URL.Path, head.Stream, start))
	}
}

// notFound answers a request for a model no online node serves: 503 with
// Retry-After if an offline peer has it (agents should retry), else 404.
func (s *Server) notFound(w http.ResponseWriter, r *http.Request, model string) {
	if r.Header.Get(cluster.HopHeader) == "" {
		if peer, seen, ok := s.cluster.FindOffline(model); ok {
			msg := fmt.Sprintf("model %q is on node %s, which is offline", model, peer)
			if !seen.IsZero() {
				msg += fmt.Sprintf(" (last seen %s ago)", time.Since(seen).Round(time.Second))
			}
			w.Header().Set("Retry-After", "10")
			s.apiError(w, r, http.StatusServiceUnavailable, "api_error", msg)
			return
		}
	}
	s.apiError(w, r, http.StatusNotFound, "not_found_error", fmt.Sprintf("model %q is not available on any node", model))
}

func (s *Server) proxy(w http.ResponseWriter, r *http.Request, body []byte, target *url.URL, hop bool) {
	rp := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(target)
			pr.Out.Host = target.Host
			pr.Out.Body = io.NopCloser(bytes.NewReader(body))
			pr.Out.ContentLength = int64(len(body))
			pr.Out.Header.Set("Content-Length", strconv.Itoa(len(body)))
			if hop {
				pr.Out.Header.Set(cluster.HopHeader, "1")
			}
		},
		Transport:     s.transport,
		FlushInterval: -1, // stream SSE tokens as they arrive
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if hop {
				var op *net.OpError
				if errors.As(err, &op) && op.Op == "dial" {
					s.cluster.MarkDown(target, err)
				}
				w.Header().Set("Retry-After", "10")
				s.apiError(w, r, http.StatusBadGateway, "api_error", "peer "+target.Host+" unreachable: "+err.Error())
				return
			}
			s.apiError(w, r, http.StatusBadGateway, "api_error", "upstream: "+err.Error())
		},
	}
	rp.ServeHTTP(w, r)
}

// rewriteModel replaces the top-level "model" field of a JSON body.
func rewriteModel(body []byte, model string) []byte {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		return body
	}
	obj["model"], _ = json.Marshal(model)
	out, err := json.Marshal(obj)
	if err != nil {
		return body
	}
	return out
}

type modelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
	// Extra fields for the UI; OpenAI clients ignore them.
	Node   string          `json:"node"`
	Local  bool            `json:"local"`
	Status scheduler.State `json:"status"`
	Engine string          `json:"engine"`
	// ContextLength is the configured context, when set (vLLM/OpenRouter-style field).
	ContextLength int `json:"context_length,omitempty"`
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": s.modelList(r)})
}

// modelList is every runnable model on this node and, unless r is a hop from
// a peer, on every online peer. Never nil.
func (s *Server) modelList(r *http.Request) []modelEntry {
	var data []modelEntry
	seen := map[string]bool{}
	add := func(node string, local bool, st scheduler.Status) {
		if !st.Runnable {
			return
		}
		id := st.Name
		if seen[id] {
			id = st.Name + "@" + node // same name on two nodes: keep both addressable
		}
		seen[id] = true
		data = append(data, modelEntry{
			ID: id, Object: "model", OwnedBy: node,
			Node: node, Local: local, Status: st.State, Engine: st.Engine,
			ContextLength: st.Ctx,
		})
	}
	for _, st := range s.mgr.Snapshot() {
		add(s.cfg.Node.Name, true, st)
	}
	if r.Header.Get(cluster.HopHeader) == "" {
		for _, p := range s.cluster.Peers() {
			if p.Online && p.Node != nil {
				for _, st := range p.Node.Models {
					add(p.Name, false, st)
				}
			}
		}
	}
	if data == nil {
		data = []modelEntry{}
	}
	return data
}

func (s *Server) nodeInfo() cluster.NodeInfo {
	return cluster.NodeInfo{
		Name: s.cfg.Node.Name, Version: version.Version, OS: runtime.GOOS, Arch: runtime.GOARCH,
		MemoryBudgetGB: s.cfg.Node.MemoryBudgetGB,
		MemoryUsedGB:   s.mgr.UsedGB(),
		Role:           s.cfg.Node.Role,
		Models:         s.mgr.Snapshot(),
		Lock:           s.currentLock(),
		Hardware:       hardware(),
	}
}

// hardware is this machine's description (hostinfo caches it; tests can stub it).
var hardware = func() *hostinfo.Info {
	h := hostinfo.Get()
	return &h
}

func (s *Server) handleNode(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.nodeInfo())
}

func (s *Server) handleStatus(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"node":  s.nodeInfo(),
		"peers": s.cluster.Peers(),
	})
}

// handleLoad / handleUnload act on a local model, or forward "model@peer".
func (s *Server) handleLoad(w http.ResponseWriter, r *http.Request) {
	s.manage(w, r, "load")
}

func (s *Server) handleUnload(w http.ResponseWriter, r *http.Request) {
	s.manage(w, r, "unload")
}

func (s *Server) manage(w http.ResponseWriter, r *http.Request, action string) {
	rt, ok := s.locate(r.PathValue("name"), r.Header.Get(cluster.HopHeader) == "")
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown model"})
		return
	}
	if rt.peer != nil {
		r.URL.Path = "/api/models/" + url.PathEscape(rt.remote) + "/" + action
		s.proxy(w, r, nil, rt.peer, true)
		s.cluster.Refresh(rt.peer)
		return
	}
	allowed, done, held := s.lockGate(r)
	if !allowed {
		s.reservedError(w, r, held)
		return
	}
	defer done()
	var err error
	if action == "load" {
		err = s.mgr.Load(r.Context(), rt.local)
	} else {
		err = s.mgr.Unload(r.Context(), rt.local)
	}
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request) {
	rt, ok := s.locate(r.PathValue("name"), r.Header.Get(cluster.HopHeader) == "")
	if !ok {
		http.Error(w, "unknown model", http.StatusNotFound)
		return
	}
	if rt.peer != nil {
		r.URL.Path = "/api/models/" + url.PathEscape(rt.remote) + "/logs"
		s.proxy(w, r, nil, rt.peer, true)
		return
	}
	inst, _ := s.mgr.Get(rt.local)
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Write(inst.Logs.Bytes())
}

// apiError writes an error in the shape the calling client expects.
func (s *Server) apiError(w http.ResponseWriter, r *http.Request, code int, typ, msg string) {
	if strings.HasPrefix(r.URL.Path, "/v1/messages") {
		writeJSON(w, code, map[string]any{
			"type":  "error",
			"error": map[string]string{"type": typ, "message": msg},
		})
		return
	}
	writeJSON(w, code, map[string]any{
		"error": map[string]string{"type": typ, "message": msg},
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

type statusRecorder struct {
	http.ResponseWriter
	code int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.code = code
	r.ResponseWriter.WriteHeader(code)
}

// Flush keeps streaming working through the logging wrapper.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (r *statusRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

func (s *Server) logRequests(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/v1/") {
			h.ServeHTTP(w, r)
			return
		}
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, code: http.StatusOK}
		h.ServeHTTP(rec, r)
		s.log.Info("request", "method", r.Method, "path", r.URL.Path,
			"status", rec.code, "dur", time.Since(start).Round(time.Millisecond),
			"forwarded", r.Header.Get(cluster.HopHeader) != "")
	})
}

// meter wraps a ResponseWriter to time the response and keep its tail, from
// which token usage and backend timings are parsed afterwards.
type meter struct {
	http.ResponseWriter
	start, first, last time.Time
	code               int
	tail               []byte
}

const meterTail = 32 << 10

func (m *meter) WriteHeader(code int) {
	m.code = code
	m.ResponseWriter.WriteHeader(code)
}

func (m *meter) Write(b []byte) (int, error) {
	now := time.Now()
	if m.first.IsZero() && len(b) > 0 {
		m.first = now
	}
	m.last = now
	m.tail = append(m.tail, b...)
	if len(m.tail) > meterTail {
		m.tail = append(m.tail[:0], m.tail[len(m.tail)-meterTail:]...)
	}
	return m.ResponseWriter.Write(b)
}

func (m *meter) Flush() {
	if f, ok := m.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (m *meter) Unwrap() http.ResponseWriter { return m.ResponseWriter }

func (m *meter) sample(path string, stream bool, requested time.Time) stats.Sample {
	s := stats.Sample{At: requested, Path: path, Stream: stream, Status: m.code}
	if m.first.IsZero() {
		return s
	}
	s.TTFTSeconds = m.first.Sub(m.start).Seconds()
	s.TotalSeconds = m.last.Sub(requested).Seconds()
	if m.code < 400 {
		stats.ParseUsage(m.tail, &s)
		// No backend timings: estimate from the streaming window.
		if s.GenTPS == 0 && stream && s.CompletionTokens > 1 {
			if d := m.last.Sub(m.first).Seconds(); d > 0 {
				s.GenTPS, s.GenTPSSource = float64(s.CompletionTokens-1)/d, "measured"
			}
		}
	}
	return s
}

// modelStats is the payload of GET /api/models/{name}/stats.
type modelStats struct {
	Node     string           `json:"node"`
	Status   scheduler.Status `json:"status"`
	Config   modelConfig      `json:"config"`
	RSSBytes int64            `json:"rss_bytes,omitempty"`
	Backend  *stats.Backend   `json:"backend,omitempty"`
	GPUs     []stats.GPU      `json:"gpus,omitempty"`
	Requests stats.Summary    `json:"requests"`
}

type modelConfig struct {
	Model      string   `json:"model,omitempty"`
	Ctx        int      `json:"ctx,omitempty"`
	MemoryGB   float64  `json:"memory_gb"`
	TTLSeconds float64  `json:"ttl_seconds"`
	Args       []string `json:"args,omitempty"`
	BudgetGB   float64  `json:"node_budget_gb"`
	NodeUsedGB float64  `json:"node_used_gb"`
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	rt, ok := s.locate(r.PathValue("name"), r.Header.Get(cluster.HopHeader) == "")
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown model"})
		return
	}
	if rt.peer != nil {
		r.URL.Path = "/api/models/" + url.PathEscape(rt.remote) + "/stats"
		s.proxy(w, r, nil, rt.peer, true)
		return
	}
	m := s.cfg.Models[rt.local]
	out := modelStats{
		Node: s.cfg.Node.Name,
		Config: modelConfig{
			Model: m.Model, Ctx: m.Ctx, MemoryGB: m.MemoryGB, Args: m.Args,
			TTLSeconds: m.TTL.D().Seconds(), BudgetGB: s.cfg.Node.MemoryBudgetGB, NodeUsedGB: s.mgr.UsedGB(),
		},
		GPUs:     stats.GPUs(),
		Requests: s.stats.Summary(rt.local),
	}
	if out.Config.Model == "" && m.Cmd != "" {
		out.Config.Model = m.Cmd
	}
	for _, st := range s.mgr.Snapshot() {
		if st.Name == rt.local {
			out.Status = st
		}
	}
	if out.Status.State == scheduler.Ready {
		out.RSSBytes = stats.RSSBytes(out.Status.PID)
		out.Backend = stats.ProbeBackend(r.Context(), out.Status.Engine, out.Status.Port)
	}
	writeJSON(w, http.StatusOK, out)
}
