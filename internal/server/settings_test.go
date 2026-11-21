package server

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/hofftodd/ai-swap/internal/config"
	"github.com/hofftodd/ai-swap/internal/scheduler"
)

const settingsCfg = `
node: {name: box, memory_budget_gb: 40, resident: [a], port_range: [19600, 19649]}
models:
  a: {engine: generic, cmd: "a-backend --port ${PORT}", memory_gb: 10}
  b: {engine: generic, cmd: "b-backend --port ${PORT}", memory_gb: 10, aliases: [bee]}
  c: {engine: generic, cmd: "c-backend --port ${PORT}", memory_gb: 30}
`

// withState makes n persist its settings to a state file in a temp dir,
// as main does with state.yaml next to config.yaml.
func withState(t *testing.T, n *node) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), config.StateFileName)
	n.srv.EnableSettings(path, BaseState(n.cfg), config.State{})
	return path
}

func api(t *testing.T, method, url, token, body string) (int, map[string]any) {
	t.Helper()
	req, _ := http.NewRequest(method, url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set(LockHeader, token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	var out map[string]any
	_ = json.Unmarshal(b, &out)
	return resp.StatusCode, out
}

func strs(v any) []string {
	out := []string{}
	for _, x := range v.([]any) {
		out = append(out, x.(string))
	}
	return out
}

func waitState(t *testing.T, mgr *scheduler.Manager, name string, want scheduler.State) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		for _, st := range mgr.Snapshot() {
			if st.Name == name && st.State == want {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("%s never became %s: %+v", name, want, mgr.Snapshot())
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestSettingsAPI(t *testing.T) {
	box := newNode(t, settingsCfg)
	path := withState(t, box)

	code, out := api(t, "GET", box.url+"/api/settings", "", "")
	if code != 200 || !reflect.DeepEqual(strs(out["resident"]), []string{"a"}) || out["effective_default"] != "a" ||
		out["default_model"] != "" || out["persisted"] != true {
		t.Fatalf("initial settings: %d %v", code, out)
	}

	// Making a model resident loads it and persists the change.
	code, out = api(t, "POST", box.url+"/api/models/bee/resident", "", "")
	if code != 200 || !reflect.DeepEqual(strs(out["resident"]), []string{"a", "b"}) || !reflect.DeepEqual(strs(out["loading"]), []string{"b"}) {
		t.Fatalf("add resident: %d %v", code, out)
	}
	waitState(t, box.mgr, "b", scheduler.Ready)
	st, err := config.LoadState(path)
	if err != nil || st.Resident == nil || !reflect.DeepEqual(*st.Resident, []string{"a", "b"}) || st.DefaultModel != nil {
		t.Fatalf("not persisted: %+v %v", st, err)
	}

	// Validation: unknown models, and residents over the memory budget.
	if code, out := api(t, "PUT", box.url+"/api/settings", "", `{"resident":["a","nope"]}`); code != 400 {
		t.Fatalf("unknown resident: %d %v", code, out)
	}
	if code, out := api(t, "PUT", box.url+"/api/settings", "", `{"resident":["a","b","c"]}`); code != 400 ||
		!strings.Contains(out["error"].(string), "budget") {
		t.Fatalf("over budget: %d %v", code, out)
	}
	if code, _ := api(t, "PUT", box.url+"/api/settings", "", `{"default_model":"nope"}`); code != 400 {
		t.Fatalf("unknown default accepted: %d", code)
	}
	if code, _ := api(t, "POST", box.url+"/api/models/nope/default", "", ""); code != 404 {
		t.Fatalf("default for unknown model: %d", code)
	}

	// The default model serves requests without a model, or for "default".
	if code, out := api(t, "POST", box.url+"/api/models/b/default", "", ""); code != 200 || out["default_model"] != "b" {
		t.Fatalf("set default: %d %v", code, out)
	}
	for _, body := range []string{`{"messages":[]}`, `{"model":"default","messages":[]}`} {
		if code, out := post(t, box.url+"/v1/chat/completions", body); code != 200 || out["backend"] != "b-backend" || out["model"] != "b" {
			t.Fatalf("%s: %d %v", body, code, out)
		}
	}

	// Removing a resident leaves it loaded (it goes by its TTL).
	code, out = api(t, "DELETE", box.url+"/api/models/b/resident", "", "")
	if code != 200 || !reflect.DeepEqual(strs(out["resident"]), []string{"a"}) || box.mgr.IsResident("b") {
		t.Fatalf("remove resident: %d %v", code, out)
	}
	waitState(t, box.mgr, "b", scheduler.Ready)

	// A restart reads the state file over the config.
	st, _ = config.LoadState(path)
	cfg, _ := config.Parse([]byte(settingsCfg))
	if w := cfg.ApplyState(st); len(w) != 0 || !reflect.DeepEqual(cfg.Node.Resident, []string{"a"}) || cfg.Node.DefaultModel != "b" {
		t.Fatalf("restart view: %v %v %q", w, cfg.Node.Resident, cfg.Node.DefaultModel)
	}

	// No residents at all, and the default cleared: a request without a model is refused again.
	if code, out := api(t, "PUT", box.url+"/api/settings", "", `{"resident":[],"default_model":""}`); code != 200 ||
		len(strs(out["resident"])) != 0 || out["effective_default"] != "" {
		t.Fatalf("clear: %d %v", code, out)
	}
	if code, _ := post(t, box.url+"/v1/chat/completions", `{"messages":[]}`); code != 400 {
		t.Fatalf("no default: %d", code)
	}

	// DELETE goes back to config.yaml and removes the state file.
	code, out = api(t, "DELETE", box.url+"/api/settings", "", "")
	if code != 200 || !reflect.DeepEqual(strs(out["resident"]), []string{"a"}) || out["effective_default"] != "a" {
		t.Fatalf("reset: %d %v", code, out)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("state file left after reset: %v", err)
	}
}

func TestSettingsHonourReservation(t *testing.T) {
	box := newNode(t, settingsCfg)
	withState(t, box)
	code, out := post(t, box.url+"/api/lock", `{"owner":"ai-bench","ttl_seconds":60}`)
	if code != 200 {
		t.Fatalf("lock: %d %v", code, out)
	}
	token := out["token"].(string)

	if code, out := api(t, "POST", box.url+"/api/models/c/resident", "", ""); code != 503 || box.mgr.IsResident("c") {
		t.Fatalf("resident change while reserved by someone else: %d %v", code, out)
	}
	if code, _ := api(t, "PUT", box.url+"/api/settings", "", `{"resident":["b"]}`); code != 503 {
		t.Fatalf("PUT resident while reserved: %d", code)
	}
	// The default doesn't load or unload anything, so it can change.
	if code, out := api(t, "PUT", box.url+"/api/settings", "", `{"default_model":"b"}`); code != 200 || out["default_model"] != "b" {
		t.Fatalf("default while reserved: %d %v", code, out)
	}
	// The holder may change residents.
	if code, out := api(t, "POST", box.url+"/api/models/b/resident", token, ""); code != 200 {
		t.Fatalf("holder: %d %v", code, out)
	}
	waitState(t, box.mgr, "b", scheduler.Ready)
}

func TestSettingsOnPeer(t *testing.T) {
	linux := newNode(t, linuxCfg)
	path := withState(t, linux)
	mac := newNode(t, macCfg, linux.url)
	withState(t, mac)

	// model@node acts on that machine, as load and unload do.
	if code, out := api(t, "POST", mac.url+"/api/models/coder@linux/resident", "", ""); code != 200 ||
		out["node"] != "linux" || !reflect.DeepEqual(strs(out["resident"]), []string{"coder"}) {
		t.Fatalf("peer resident: %d %v", code, out)
	}
	if !linux.mgr.IsResident("coder") || mac.mgr.IsResident("coder") {
		t.Fatal("resident set on the wrong machine")
	}
	if st, _ := config.LoadState(path); st.Resident == nil || (*st.Resident)[0] != "coder" {
		t.Fatalf("peer didn't persist: %+v", st)
	}

	// ?node= targets a machine for the whole settings document.
	if code, out := api(t, "PUT", mac.url+"/api/settings?node=linux", "", `{"default_model":"c"}`); code != 200 || out["default_model"] != "coder" {
		t.Fatalf("peer PUT: %d %v", code, out)
	}
	if code, out := api(t, "GET", mac.url+"/api/settings?node=linux", "", ""); code != 200 || out["node"] != "linux" || out["default_model"] != "coder" {
		t.Fatalf("peer GET: %d %v", code, out)
	}
	if code, _ := api(t, "GET", mac.url+"/api/settings?node=nowhere", "", ""); code != 404 {
		t.Fatalf("unknown node: %d", code)
	}

	// A node's default can be a peer's model.
	if code, out := api(t, "PUT", mac.url+"/api/settings", "", `{"default_model":"c@linux"}`); code != 200 || out["default_model"] != "coder@linux" {
		t.Fatalf("peer default: %d %v", code, out)
	}
	if code, out := post(t, mac.url+"/v1/chat/completions", `{"messages":[]}`); code != 200 || out["backend"] != "coder-backend" {
		t.Fatalf("routed to peer default: %d %v", code, out)
	}
}
