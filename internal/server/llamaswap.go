package server

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"

	"github.com/hofftodd/ai-swap/internal/scheduler"
)

// llama-swap compatibility, so tools written for llama-swap (benchmark
// harnesses, scripts that free the GPU first) work against ai-swap unchanged.
// These act on this machine's models only, like llama-swap on one machine:
//
//	GET  /running                    models loaded or loading here
//	GET|POST /unload                 unload every model here (also POST /api/unload)
//	GET  /upstream/{model}/{path...} pass a GET straight to that model's engine
//
// Unloading by hand keeps a resident model unloaded until something uses it
// again, so a benchmark gets the whole GPU.

type running struct {
	Model string `json:"model"`
	State string `json:"state"`
	Proxy string `json:"proxy"`
	Cmd   string `json:"cmd,omitempty"`
}

func (s *Server) handleRunning(w http.ResponseWriter, _ *http.Request) {
	out := []running{}
	for _, st := range s.mgr.Snapshot() {
		if st.State != scheduler.Ready && st.State != scheduler.Starting {
			continue
		}
		out = append(out, running{
			Model: st.Name, State: string(st.State), Cmd: st.Cmd,
			Proxy: "http://127.0.0.1:" + strconv.Itoa(st.Port),
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"running": out})
}

func (s *Server) handleUnloadAll(w http.ResponseWriter, r *http.Request) {
	allowed, done, held := s.lockGate(r)
	if !allowed {
		s.reservedError(w, r, held)
		return
	}
	defer done()
	var errs []error
	for _, st := range s.mgr.Snapshot() {
		if st.State == scheduler.Stopped || st.State == scheduler.Failed {
			continue
		}
		if err := s.mgr.Unload(r.Context(), st.Name); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", st.Name, err))
		}
	}
	if err := errors.Join(errs...); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	io.WriteString(w, "OK")
}

func (s *Server) handleUpstream(w http.ResponseWriter, r *http.Request) {
	m, ok := s.cfg.Resolve(r.PathValue("name"))
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown model"})
		return
	}
	allowed, done, held := s.lockGate(r)
	if !allowed {
		s.reservedError(w, r, held)
		return
	}
	defer done()
	var port int
	for _, st := range s.mgr.Snapshot() {
		if st.Name == m.Name && st.State == scheduler.Ready {
			port = st.Port
		}
	}
	if port == 0 {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": m.Name + " is not loaded"})
		return
	}
	target := &url.URL{Scheme: "http", Host: "127.0.0.1:" + strconv.Itoa(port)}
	path := "/" + strings.TrimPrefix(r.PathValue("path"), "/")
	rp := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(target)
			pr.Out.URL.Path, pr.Out.URL.RawPath = path, ""
			pr.Out.Host = target.Host
		},
		Transport: s.transport,
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			writeJSON(w, http.StatusBadGateway, map[string]string{"error": "upstream: " + err.Error()})
		},
	}
	rp.ServeHTTP(w, r)
}
