package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/hofftodd/ai-swap/internal/cluster"
	"github.com/hofftodd/ai-swap/internal/config"
)

// Node settings changed at runtime: which models are resident and which is
// the default. They are kept in a state file next to config.yaml (see
// config.State) and laid over the config at startup, so the config file is
// never rewritten and a tool that generates it (llama-config) can't lose them.
type settingsStore struct {
	mu   sync.Mutex // serialises read-modify-write of the state file
	path string     // "" keeps changes in memory only
	base config.State
	over config.State
}

// EnableSettings sets where runtime settings persist. base is what the config
// file says (before the state was applied), over is the loaded state file.
func (s *Server) EnableSettings(path string, base, over config.State) {
	s.settings.mu.Lock()
	defer s.settings.mu.Unlock()
	s.settings.path, s.settings.base, s.settings.over = path, base, over
}

// BaseState is the config file's resident set and default, as a State.
func BaseState(cfg *config.Config) config.State {
	res := append([]string{}, cfg.Node.Resident...)
	def := cfg.Node.DefaultModel
	return config.State{Resident: &res, DefaultModel: &def}
}

// settingsView is the payload of GET/PUT /api/settings.
type settingsView struct {
	Node string `json:"node"`
	// Resident and DefaultModel are in effect now; EffectiveDefault is the
	// model a request without a name gets (DefaultModel, else the first resident).
	Resident         []string `json:"resident"`
	DefaultModel     string   `json:"default_model"`
	EffectiveDefault string   `json:"effective_default"`
	// Config is what config.yaml says; Overrides is what the state file
	// changes (nil fields are not overridden).
	Config    config.State `json:"config"`
	Overrides config.State `json:"overrides"`
	Persisted bool         `json:"persisted"`
	// Loading lists new residents being loaded in the background (PUT only).
	Loading []string `json:"loading,omitempty"`
}

func (s *Server) settingsViewLocked() settingsView {
	return settingsView{
		Node: s.cfg.Node.Name, Resident: s.mgr.Residents(),
		DefaultModel: s.mgr.ConfiguredDefault(), EffectiveDefault: s.mgr.DefaultModel(),
		Config: s.settings.base, Overrides: s.settings.over, Persisted: s.settings.path != "",
	}
}

// forwardNode sends a ?node=<peer> request to that peer. It reports whether
// the request was handled (forwarded, or answered with an error).
func (s *Server) forwardNode(w http.ResponseWriter, r *http.Request) bool {
	n := r.URL.Query().Get("node")
	if n == "" || n == s.cfg.Node.Name {
		return false
	}
	if r.Header.Get(cluster.HopHeader) == "" {
		for _, p := range s.cluster.Peers() {
			if p.Name == n || (p.Node != nil && p.Node.Name == n) {
				if u, err := url.Parse(p.URL); err == nil {
					q := r.URL.Query()
					q.Del("node")
					r.URL.RawQuery = q.Encode()
					body, _ := readSmall(w, r)
					s.proxy(w, r, body, u, true)
					s.cluster.Refresh(u)
					return true
				}
			}
		}
	}
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown node " + n})
	return true
}

func readSmall(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}
	return io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<16))
}

// GET /api/settings[?node=<peer>]
func (s *Server) handleSettings(w http.ResponseWriter, r *http.Request) {
	if s.forwardNode(w, r) {
		return
	}
	s.settings.mu.Lock()
	defer s.settings.mu.Unlock()
	writeJSON(w, http.StatusOK, s.settingsViewLocked())
}

// settingsChange is the body of PUT /api/settings. Absent fields are left
// as they are; "default_model": "" clears the default (the first resident
// is used); "resident": [] means no residents.
type settingsChange struct {
	Resident     *[]string `json:"resident"`
	DefaultModel *string   `json:"default_model"`
}

// PUT /api/settings[?node=<peer>]
func (s *Server) handlePutSettings(w http.ResponseWriter, r *http.Request) {
	if s.forwardNode(w, r) {
		return
	}
	var ch settingsChange
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&ch); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "body must be JSON: " + err.Error()})
		return
	}
	s.changeSettings(w, r, ch, false)
}

// DELETE /api/settings[?node=<peer>]: drop the overrides, back to config.yaml.
func (s *Server) handleResetSettings(w http.ResponseWriter, r *http.Request) {
	if s.forwardNode(w, r) {
		return
	}
	s.settings.mu.Lock()
	base := s.settings.base
	s.settings.mu.Unlock()
	s.changeSettings(w, r, settingsChange{Resident: base.Resident, DefaultModel: base.DefaultModel}, true)
}

// POST|DELETE /api/models/{name}/resident and POST /api/models/{name}/default
// change one model's setting on the machine that has it ("model@node"
// forwards to that peer, as load and unload do).
func (s *Server) handleModelSetting(w http.ResponseWriter, r *http.Request) {
	what := "resident"
	if strings.HasSuffix(r.URL.Path, "/default") {
		what = "default"
	}
	rt, ok := s.locate(r.PathValue("name"), r.Header.Get(cluster.HopHeader) == "")
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown model"})
		return
	}
	if rt.peer != nil {
		r.URL.Path = "/api/models/" + url.PathEscape(rt.remote) + "/" + what
		s.proxy(w, r, nil, rt.peer, true)
		s.cluster.Refresh(rt.peer)
		return
	}
	if what == "default" {
		d := rt.local
		s.changeSettings(w, r, settingsChange{DefaultModel: &d}, false)
		return
	}
	res := s.mgr.Residents()
	var next []string
	for _, n := range res {
		if n != rt.local {
			next = append(next, n)
		}
	}
	if r.Method == http.MethodPost {
		next = append(next, rt.local)
	}
	if next == nil {
		next = []string{}
	}
	s.changeSettings(w, r, settingsChange{Resident: &next}, false)
}

// changeSettings validates, persists and applies a change. Changing the
// resident set honours a reservation like load and unload do: only its holder
// may. New residents are loaded in the background (evicting idle models if
// the memory budget needs it); a model that stops being resident is unloaded
// by its idle TTL. reset drops the state file's overrides.
func (s *Server) changeSettings(w http.ResponseWriter, r *http.Request, ch settingsChange, reset bool) {
	s.settings.mu.Lock()
	defer s.settings.mu.Unlock()

	var res []string
	if ch.Resident != nil {
		var err error
		if res, err = s.cfg.CheckResident(*ch.Resident); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "resident: " + err.Error()})
			return
		}
	}
	var def string
	if ch.DefaultModel != nil {
		def = strings.TrimSpace(*ch.DefaultModel)
		if def != "" {
			d, err := s.checkDefault(r, def)
			if err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": "default_model: " + err.Error()})
				return
			}
			def = d
		}
	}

	var token string
	if ch.Resident != nil {
		allowed, done, held := s.lockGate(r)
		if !allowed {
			s.reservedError(w, r, held)
			return
		}
		done()
		token = r.Header.Get(LockHeader)
	}

	over := s.settings.over
	if reset {
		over = config.State{}
	} else {
		if ch.Resident != nil {
			over.Resident = &res
		}
		if ch.DefaultModel != nil {
			over.DefaultModel = &def
		}
	}
	if s.settings.path != "" {
		if err := config.SaveState(s.settings.path, over); err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "save settings: " + err.Error()})
			return
		}
	}
	s.settings.over = over

	var loading []string
	if ch.Resident != nil {
		before := s.mgr.Residents()
		s.mgr.SetResident(res)
		for _, n := range res {
			if contains(before, n) {
				continue
			}
			if inst, ok := s.mgr.Get(n); ok && inst.Engine.Supported() {
				loading = append(loading, n)
				go s.loadResident(n, token)
			}
		}
		s.log.Info("resident models changed", "resident", res)
	}
	if ch.DefaultModel != nil {
		s.mgr.SetDefault(def)
		s.log.Info("default model changed", "default", def)
	}
	v := s.settingsViewLocked()
	v.Loading = loading
	writeJSON(w, http.StatusOK, v)
}

// loadResident loads a model just made resident, unless the machine has since
// been reserved by someone other than whoever made the change.
func (s *Server) loadResident(name, token string) {
	if l := s.currentLock(); l != nil {
		s.lock.mu.Lock()
		mine := token != "" && token == s.lock.token
		s.lock.mu.Unlock()
		if !mine {
			return // the reaper restores it once the reservation ends
		}
	}
	if err := s.mgr.Load(context.Background(), name); err != nil {
		s.log.Warn("loading new resident failed", "model", name, "err", err)
	}
}

// checkDefault canonicalises a default model: a local model or alias, or
// "model@node" naming a peer's model that is online now.
func (s *Server) checkDefault(r *http.Request, name string) (string, error) {
	want, node, pinned := strings.Cut(name, "@")
	if !pinned || node == s.cfg.Node.Name {
		if m, ok := s.cfg.Resolve(want); ok {
			return m.Name, nil
		}
		return "", fmt.Errorf("unknown model %q", name)
	}
	if r.Header.Get(cluster.HopHeader) == "" {
		if _, remote, ok := s.cluster.Find(name); ok {
			return remote + "@" + node, nil
		}
	}
	return "", fmt.Errorf("no online node has %q", name)
}

// defaultName maps a request's model to the node's default when it names none
// or asks for "default" (and no model is called that).
func (s *Server) defaultName(model string) string {
	if model == "" || model == "default" {
		if _, ok := s.cfg.Resolve(model); !ok || model == "" {
			if d := s.mgr.DefaultModel(); d != "" {
				return d
			}
		}
	}
	return model
}

func contains(list []string, x string) bool {
	for _, v := range list {
		if v == x {
			return true
		}
	}
	return false
}
