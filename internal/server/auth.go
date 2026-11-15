package server

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// requireKey checks node.api_key on every API route: "Authorization: Bearer
// <key>" (OpenAI clients, peers) or "x-api-key: <key>" (Anthropic clients).
// The UI's static files and /health stay open; the UI itself asks for the key
// and sends it like any client. With no key configured, everything is open.
func (s *Server) requireKey(h http.Handler) http.Handler {
	key := s.cfg.Node.APIKey
	if key == "" {
		return h
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !protected(r.URL.Path) || validKey(r, key) {
			h.ServeHTTP(w, r)
			return
		}
		w.Header().Set("WWW-Authenticate", `Bearer realm="ai-swap"`)
		s.apiError(w, r, http.StatusUnauthorized, "authentication_error",
			"missing or wrong API key (send Authorization: Bearer <key> or x-api-key)")
	})
}

// protected reports whether a path needs the key: the inference and
// management APIs and the llama-swap compatible routes, not the UI.
func protected(path string) bool {
	for _, p := range []string{"/v1/", "/api/", "/upstream/"} {
		if strings.HasPrefix(path, p) {
			return true
		}
	}
	return path == "/running" || path == "/unload"
}

func validKey(r *http.Request, key string) bool {
	got := r.Header.Get("x-api-key")
	if a := r.Header.Get("Authorization"); got == "" && len(a) > 7 && strings.EqualFold(a[:7], "bearer ") {
		got = strings.TrimSpace(a[7:])
	}
	return got != "" && subtle.ConstantTimeCompare([]byte(got), []byte(key)) == 1
}
