package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/hofftodd/ai-swap/internal/scheduler"
)

// Two models that can't both fit in the budget.
const leaseCfg = `
node: {name: box, memory_budget_gb: 10, port_range: [19400, 19449]}
models:
  small:
    engine: generic
    cmd: "small-backend --port ${PORT}"
    memory_gb: 6
  big:
    engine: generic
    cmd: "big-backend --port ${PORT}"
    memory_gb: 6
`

func send(t *testing.T, method, url, body string) (int, map[string]any) {
	t.Helper()
	req, _ := http.NewRequest(method, url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
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

func state(n *node, model string) scheduler.State {
	for _, s := range n.mgr.Snapshot() {
		if s.Name == model {
			return s.State
		}
	}
	return ""
}

func TestLeaseKeepsModelLoaded(t *testing.T) {
	n := newNode(t, leaseCfg)

	code, out := send(t, "POST", n.url+"/api/leases", `{"model":"small","owner":"ai-factory","reason":"builder step","ttl_seconds":60}`)
	if code != 200 {
		t.Fatalf("lease: %d %v", code, out)
	}
	id, _ := out["lease_id"].(string)
	if id == "" || out["node"] != "box" {
		t.Fatalf("lease reply: %v", out)
	}
	if state(n, "small") != scheduler.Ready {
		t.Fatalf("leasing should load the model, got %s", state(n, "small"))
	}

	// The node shows the lease, without its id.
	_, node := send(t, "GET", n.url+"/api/node", "")
	b, _ := json.Marshal(node["leases"])
	if !strings.Contains(string(b), `"owner":"ai-factory"`) || strings.Contains(string(b), id) {
		t.Fatalf("node leases: %s", b)
	}

	// A load that needs small's memory waits instead of evicting it.
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if err := n.mgr.Load(ctx, "big"); err == nil {
		t.Fatal("big loaded while small was leased")
	}
	if state(n, "small") != scheduler.Ready {
		t.Fatalf("small was evicted: %s", state(n, "small"))
	}

	// Renewing works; an unknown id is 404.
	if code, out := send(t, "POST", n.url+"/api/leases", `{"lease_id":"`+id+`","ttl_seconds":120}`); code != 200 {
		t.Fatalf("renew: %d %v", code, out)
	}
	if code, _ := send(t, "POST", n.url+"/api/leases", `{"lease_id":"nope"}`); code != 404 {
		t.Fatalf("renew unknown: %d", code)
	}

	// After release, big can load and evicts small.
	if code, out := send(t, "DELETE", n.url+"/api/leases/"+id, ""); code != 200 || out["released"] != true {
		t.Fatalf("release: %d %v", code, out)
	}
	ctx2, cancel2 := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel2()
	if err := n.mgr.Load(ctx2, "big"); err != nil {
		t.Fatalf("big after release: %v", err)
	}
	if state(n, "small") == scheduler.Ready {
		t.Fatal("small should have been evicted for big")
	}
}

func TestLeaseValidationAndLapse(t *testing.T) {
	n := newNode(t, leaseCfg)
	if code, _ := send(t, "POST", n.url+"/api/leases", `{"owner":"x"}`); code != 400 {
		t.Fatalf("no model: %d", code)
	}
	if code, _ := send(t, "POST", n.url+"/api/leases", `{"model":"small"}`); code != 400 {
		t.Fatalf("no owner: %d", code)
	}
	if code, _ := send(t, "POST", n.url+"/api/leases", `{"model":"missing","owner":"x"}`); code != 404 {
		t.Fatalf("unknown model: %d", code)
	}

	// A lease nobody renews lapses, and the model is free to go.
	if code, out := send(t, "POST", n.url+"/api/leases", `{"model":"small","owner":"crashed","ttl_seconds":1}`); code != 200 {
		t.Fatalf("lease: %d %v", code, out)
	}
	time.Sleep(1100 * time.Millisecond)
	if _, out := send(t, "GET", n.url+"/api/leases", ""); len(out["leases"].([]any)) != 0 {
		t.Fatalf("lease should have lapsed: %v", out)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := n.mgr.Load(ctx, "big"); err != nil {
		t.Fatalf("big after lapse: %v", err)
	}
}

func TestLeaseOnPeerModel(t *testing.T) {
	linux := newNode(t, linuxCfg)
	mac := newNode(t, macCfg, linux.url)

	code, out := send(t, "POST", mac.url+"/api/leases", `{"model":"coder@linux","owner":"ai-delegate"}`)
	if code != 200 || out["node"] != "linux" {
		t.Fatalf("lease via peer: %d %v", code, out)
	}
	id := out["lease_id"].(string)
	if state(linux, "coder") != scheduler.Ready {
		t.Fatal("peer model should be loaded")
	}
	if _, out := send(t, "GET", linux.url+"/api/leases", ""); len(out["leases"].([]any)) != 1 {
		t.Fatalf("linux should hold the lease: %v", out)
	}
	if code, _ := send(t, "POST", mac.url+"/api/leases", `{"model":"coder@linux","lease_id":"`+id+`"}`); code != 200 {
		t.Fatalf("renew via peer: %d", code)
	}
	if code, out := send(t, "DELETE", mac.url+"/api/leases/"+id+"?model=coder@linux", ""); code != 200 || out["released"] != true {
		t.Fatalf("release via peer: %d %v", code, out)
	}
	if _, out := send(t, "GET", linux.url+"/api/leases", ""); len(out["leases"].([]any)) != 0 {
		t.Fatalf("linux lease should be gone: %v", out)
	}
}

func TestStatsDescribeUnloadedModel(t *testing.T) {
	n := newNode(t, `
node: {name: box, port_range: [19450, 19499]}
models:
  m:
    engine: generic
    cmd: "m-backend --port ${PORT}"
    args: [--cache-type-k, q8_0, --cache-type-v, q8_0]
    params: 35B
    active_params: 3B
    quant: Q4_K_M
    lab: Example Lab
    url: https://huggingface.co/example/m
`)
	code, out := send(t, "GET", n.url+"/api/models/m/stats", "")
	if code != 200 {
		t.Fatalf("stats: %d %v", code, out)
	}
	f, _ := out["file"].(map[string]any)
	if f["params"] != float64(35e9) || f["active_params"] != float64(3e9) || f["quant"] != "Q4_K_M" ||
		f["lab"] != "Example Lab" || f["url"] != "https://huggingface.co/example/m" {
		t.Fatalf("file facts: %v", out["file"])
	}
	if c, _ := out["config"].(map[string]any); c["kv_cache"] != "q8_0" {
		t.Fatalf("kv cache: %v", out["config"])
	}
}

const midCfg = `
node: {name: mid, port_range: [19500, 19549]}
models:
  helper:
    engine: generic
    cmd: "helper-backend --port ${PORT}"
`

// A node configured with one peer learns that peer's peers and routes to them
// directly.
func TestPeersLearnedTransitively(t *testing.T) {
	linux := newNode(t, linuxCfg)
	mid := newNode(t, midCfg, linux.url)
	mac := newNode(t, macCfg, mid.url)

	deadline := time.Now().Add(3 * time.Second)
	for {
		code, out := post(t, mac.url+"/v1/chat/completions", `{"model":"coder","messages":[]}`)
		if code == 200 && out["backend"] == "coder-backend" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("mac never reached linux's model: %d %v", code, out)
		}
		time.Sleep(50 * time.Millisecond)
	}
	_, st := send(t, "GET", mac.url+"/api/status", "")
	b, _ := json.Marshal(st["peers"])
	if !strings.Contains(string(b), `"learned":true`) || !strings.Contains(string(b), `"name":"linux"`) {
		t.Fatalf("mac should list linux as learned: %s", b)
	}
	// mac publishes its own peers but never itself.
	_, node := send(t, "GET", mac.url+"/api/node", "")
	if b, _ := json.Marshal(node["peers"]); strings.Contains(string(b), `"name":"mac"`) {
		t.Fatalf("mac lists itself: %s", b)
	}
}
