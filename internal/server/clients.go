package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

// Client configs: GET /api/clients/{client} downloads a config that points a
// coding agent at every model in the cluster through this ai-swap endpoint.
// The output matches what the ai-swap-clients script (in ai-tools) merges into
// the agents' own files; these are standalone files, so they replace rather
// than merge.
//
//	pi        models.json    → ~/.pi/agent/models.json
//	omp       models.yml     → ~/.omp/agent/models.yml
//	opencode  opencode.json  → ~/.config/opencode/opencode.json
//	claude    claude-swap    → ~/.local/bin/claude-swap (chmod +x)
//
// Query: default=<model id> (opencode's default model and claude-swap's
// default; else the first model), endpoint=<http(s)://host:port> (else the
// address the request came in on).

const (
	clientPrefix     = "ai-swap-"
	clientDefaultCtx = 131072 // oMLX's and llama-config's default when a model doesn't state one
	clientMaxOut     = 32768
)

type clientFile struct {
	name, contentType string
	render            func(models []modelEntry, base, endpoint, def string) ([]byte, error)
}

var clientFiles = map[string]clientFile{
	"pi":       {"models.json", "application/json", renderPi},
	"omp":      {"models.yml", "application/yaml", renderOmp},
	"opencode": {"opencode.json", "application/json", renderOpencode},
	"claude":   {"claude-swap", "text/x-shellscript", renderClaude},
}

var (
	safeEndpoint = regexp.MustCompile(`^https?://[A-Za-z0-9.\-]+(:[0-9]+)?$|^https?://\[[0-9A-Fa-f:.]+\](:[0-9]+)?$`)
	unsafeKey    = regexp.MustCompile(`[^a-z0-9._-]`)
	safeShellID  = regexp.MustCompile(`^[A-Za-z0-9._@:/+-]+$`)
)

func (s *Server) handleClientConfig(w http.ResponseWriter, r *http.Request) {
	cf, ok := clientFiles[r.PathValue("client")]
	if !ok {
		http.Error(w, "unknown client; one of: pi, omp, opencode, claude", http.StatusNotFound)
		return
	}
	endpoint := strings.TrimRight(r.URL.Query().Get("endpoint"), "/")
	if endpoint == "" {
		endpoint = requestOrigin(r)
	}
	// The endpoint ends up in a shell script and in agents' configs: only a bare origin.
	if !safeEndpoint.MatchString(endpoint) {
		http.Error(w, "endpoint must be http(s)://host[:port]", http.StatusBadRequest)
		return
	}
	models := s.modelList(r)
	if len(models) == 0 {
		http.Error(w, "no models to configure (a peer may still be connecting)", http.StatusServiceUnavailable)
		return
	}
	def := r.URL.Query().Get("default")
	if def == "" {
		def = models[0].ID
	} else if !hasModel(models, def) {
		http.Error(w, fmt.Sprintf("default %q is not a model here", def), http.StatusBadRequest)
		return
	}
	body, err := cf.render(models, endpoint+"/v1", endpoint, def)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", cf.contentType+"; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="`+cf.name+`"`)
	w.Header().Set("Cache-Control", "no-store")
	w.Write(body)
}

// requestOrigin is the origin the client used to reach us, so a config
// downloaded from http://studio.local:8080 points back at studio.local.
func requestOrigin(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	if p := r.Header.Get("X-Forwarded-Proto"); p == "http" || p == "https" {
		scheme = p
	}
	host := r.Host
	if h := r.Header.Get("X-Forwarded-Host"); h != "" {
		host = strings.TrimSpace(strings.Split(h, ",")[0])
	}
	return (&url.URL{Scheme: scheme, Host: host}).String()
}

func hasModel(models []modelEntry, id string) bool {
	for _, m := range models {
		if m.ID == id {
			return true
		}
	}
	return false
}

func clientKey(id string) string {
	return clientPrefix + unsafeKey.ReplaceAllString(strings.ToLower(id), "-")
}

func clientCtx(m modelEntry) int {
	if m.ContextLength > 0 {
		return m.ContextLength
	}
	return clientDefaultCtx
}

func displayName(m modelEntry) string { return fmt.Sprintf("%s (%s)", m.ID, m.Node) }

// thinkFormat is pi's hint for how the model's chat template marks reasoning
// (same rules as llama-config and ai-swap-clients).
func thinkFormat(id string) string {
	m := strings.ToLower(id)
	switch {
	case strings.Contains(m, "qwen"), strings.Contains(m, "ornith"), strings.Contains(m, "thinkingcap"):
		return "qwen-chat-template"
	case strings.Contains(m, "deepseek"):
		return "deepseek"
	case strings.Contains(m, "glm"), strings.Contains(m, "zai"):
		return "zai"
	}
	return ""
}

func piProviders(models []modelEntry, base string, extra map[string]any) map[string]any {
	out := map[string]any{}
	for _, m := range models {
		compat := map[string]any{"supportsDeveloperRole": false, "supportsReasoningEffort": false}
		if tf := thinkFormat(m.ID); tf != "" {
			compat["thinkingFormat"] = tf
		}
		p := map[string]any{
			"baseUrl": base, "api": "openai-completions", "apiKey": "no-key-required", "compat": compat,
			"models": []any{map[string]any{
				"id": m.ID, "name": displayName(m), "reasoning": true, "input": []string{"text"},
				"contextWindow": clientCtx(m), "maxTokens": clientMaxOut,
				"cost": map[string]int{"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
			}},
		}
		for k, v := range extra {
			p[k] = v
		}
		out[clientKey(m.ID)] = p
	}
	return out
}

func jsonFile(v any) ([]byte, error) {
	b, err := json.MarshalIndent(v, "", "  ")
	return append(b, '\n'), err
}

func renderPi(models []modelEntry, base, _, _ string) ([]byte, error) {
	return jsonFile(map[string]any{"providers": piProviders(models, base, nil)})
}

func renderOmp(models []modelEntry, base, _, _ string) ([]byte, error) {
	return yaml.Marshal(map[string]any{"providers": piProviders(models, base, map[string]any{"auth": "none"})})
}

func renderOpencode(models []modelEntry, base, _, def string) ([]byte, error) {
	prov := map[string]any{}
	for _, m := range models {
		prov[clientKey(m.ID)] = map[string]any{
			"npm": "@ai-sdk/openai-compatible", "name": displayName(m),
			"options": map[string]any{"baseURL": base, "apiKey": "no-key-required"},
			"models": map[string]any{m.ID: map[string]any{
				"name":  displayName(m),
				"limit": map[string]int{"context": clientCtx(m), "output": clientMaxOut},
			}},
		}
	}
	return jsonFile(map[string]any{
		"$schema":  "https://opencode.ai/config.json",
		"model":    clientKey(def) + "/" + def,
		"provider": prov,
	})
}

func renderClaude(_ []modelEntry, _, endpoint, def string) ([]byte, error) {
	if !safeShellID.MatchString(def) {
		return nil, fmt.Errorf("model id %q can't be written into a shell script", def)
	}
	return []byte(fmt.Sprintf(`#!/usr/bin/env bash
# Generated by ai-swap. Claude Code against the ai-swap cluster at %[1]s.
# Save as ~/.local/bin/claude-swap and chmod +x it.
# Any model on any machine: claude-swap --model <id>   (ids: curl -s %[1]s/v1/models)
export ANTHROPIC_BASE_URL="%[1]s"
export ANTHROPIC_API_KEY="no-key-required"
export ANTHROPIC_MODEL="${AI_SWAP_MODEL:-%[2]s}"
export ANTHROPIC_SMALL_FAST_MODEL="${AI_SWAP_SMALL_MODEL:-$ANTHROPIC_MODEL}"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
exec claude "$@"
`, endpoint, def)), nil
}
