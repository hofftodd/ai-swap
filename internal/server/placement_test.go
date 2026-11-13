package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/hofftodd/ai-swap/internal/scheduler"
)

func TestPlace(t *testing.T) {
	local := candidate{node: "mac", local: true, memoryGB: 30, freeGB: 100}
	peer := candidate{node: "linux", memoryGB: 30, freeGB: 100}
	with := func(c candidate, f func(*candidate)) candidate { f(&c); return c }
	ready := func(c *candidate) { c.state = scheduler.Ready }
	busy := func(c *candidate) { c.state, c.inflight = scheduler.Ready, 2 }
	full := func(c *candidate) { c.freeGB = 10 }
	reserved := func(c *candidate) { c.reserved = true }

	cases := []struct {
		name  string
		cands []candidate
		want  string
	}{
		{"both cold and fit: local", []candidate{peer, local}, "mac"},
		{"loaded peer beats cold local", []candidate{local, with(peer, ready)}, "linux"},
		{"both loaded: local", []candidate{with(peer, ready), with(local, ready)}, "mac"},
		{"both loaded: less busy", []candidate{with(local, busy), with(peer, ready)}, "linux"},
		{"starting counts as loaded", []candidate{local, with(peer, func(c *candidate) {
			c.state = scheduler.Starting
		})}, "linux"},
		{"stopping does not", []candidate{with(peer, func(c *candidate) {
			c.state = scheduler.Stopping
		}), local}, "mac"},
		{"peer fits, local would swap", []candidate{with(local, full), peer}, "linux"},
		{"neither fits: local", []candidate{with(peer, full), with(local, full)}, "mac"},
		{"unknown size never fits", []candidate{with(local, full), with(peer, func(c *candidate) {
			c.memoryGB = 0
		})}, "mac"},
		{"reserved local, cold peer", []candidate{with(local, func(c *candidate) {
			ready(c)
			reserved(c)
		}), peer}, "linux"},
		{"all reserved: still answers", []candidate{with(peer, reserved), with(local, reserved)}, "mac"},
		{"peer only", []candidate{peer}, "linux"},
	}
	for _, tc := range cases {
		got, ok := place(tc.cands)
		if !ok || got.node != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got.node, tc.want)
		}
	}
	if _, ok := place(nil); ok {
		t.Error("place(nil) should find nothing")
	}
}

// Both machines define "shared"; each backend answers with its own command
// name, so a response shows which machine served it.
const sharedLinux = `
node: {name: linux, memory_budget_gb: 32, port_range: [19400, 19449]}
models:
  shared: {engine: generic, cmd: "linux-shared --port ${PORT}", memory_gb: 20}
`

const sharedMac = `
node: {name: mac, memory_budget_gb: 64, port_range: [19450, 19499]}
models:
  shared: {engine: generic, cmd: "mac-shared --port ${PORT}", memory_gb: 20}
  big: {engine: generic, cmd: "mac-big --port ${PORT}", memory_gb: 50}
`

func servedBy(t *testing.T, base, model string, header ...string) string {
	t.Helper()
	req, _ := http.NewRequest("POST", base+"/v1/chat/completions",
		strings.NewReader(`{"model":"`+model+`","messages":[]}`))
	req.Header.Set("Content-Type", "application/json")
	if len(header) == 2 {
		req.Header.Set(header[0], header[1])
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if resp.StatusCode != 200 {
		t.Fatalf("%s: status %d %v", model, resp.StatusCode, out)
	}
	return out["backend"].(string)
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestPlacementAcrossMachines(t *testing.T) {
	linux := newNode(t, sharedLinux)
	mac := newNode(t, sharedMac, linux.url)

	// Neither has it loaded, and it fits here: served here, as before.
	if got := servedBy(t, mac.url, "shared"); got != "mac-shared" {
		t.Fatalf("cold, fits locally: served by %s", got)
	}

	// Loaded on the peer only: the peer serves it, rather than this machine
	// loading its own copy.
	if code, _ := post(t, mac.url+"/api/models/shared/unload", ""); code != 200 {
		t.Fatal("unload failed")
	}
	if code, _ := post(t, mac.url+"/api/models/shared@peer0/load", ""); code != 200 {
		t.Fatal("peer load failed")
	}
	waitFor(t, "peer copy ready", func() bool { return peerModelReady(t, mac.url, "shared@peer0") })
	if got := servedBy(t, mac.url, "shared"); got != "linux-shared" {
		t.Fatalf("loaded on peer: served by %s", got)
	}
	if st, _ := mac.mgr.Get("shared"); st == nil {
		t.Fatal("no local instance")
	}
	for _, st := range mac.mgr.Snapshot() {
		if st.Name == "shared" && st.State != scheduler.Stopped {
			t.Fatalf("local copy was loaded anyway: %s", st.State)
		}
	}

	// Pinning still wins, and management endpoints still mean this machine.
	if got := servedBy(t, mac.url, "shared@mac"); got != "mac-shared" {
		t.Fatalf("pinned to mac: served by %s", got)
	}
	if code, _ := post(t, mac.url+"/api/models/shared/unload", ""); code != 200 {
		t.Fatal("local unload failed")
	}
	waitFor(t, "peer still ready", func() bool { return peerModelReady(t, mac.url, "shared@peer0") })
}

func TestPlacementPrefersFreeMemory(t *testing.T) {
	linux := newNode(t, sharedLinux)
	mac := newNode(t, sharedMac, linux.url)

	// "big" fills this machine (50 of 64 GB), so loading "shared" here would
	// unload it; the peer has room, so the request goes there.
	if code, _ := post(t, mac.url+"/api/models/big/load", ""); code != 200 {
		t.Fatal("big load failed")
	}
	if got := servedBy(t, mac.url, "shared"); got != "linux-shared" {
		t.Fatalf("local full, peer free: served by %s", got)
	}
	for _, st := range mac.mgr.Snapshot() {
		if st.Name == "big" && st.State != scheduler.Ready {
			t.Fatalf("big was swapped out: %s", st.State)
		}
	}
}

func TestPlacementAndReservations(t *testing.T) {
	linux := newNode(t, sharedLinux)
	mac := newNode(t, sharedMac, linux.url)

	// Reserve this machine: others are sent to the peer instead of a 503...
	code, out := post(t, mac.url+"/api/lock", `{"owner":"ai-bench","ttl_seconds":60}`)
	if code != 200 {
		t.Fatalf("lock: %d %v", code, out)
	}
	token := out["token"].(string)
	if got := servedBy(t, mac.url, "shared"); got != "linux-shared" {
		t.Fatalf("reserved here: served by %s", got)
	}
	// ...while the holder is served here, even with the peer's copy loaded.
	waitFor(t, "peer copy ready", func() bool { return peerModelReady(t, mac.url, "shared@peer0") })
	if got := servedBy(t, mac.url, "shared", LockHeader, token); got != "mac-shared" {
		t.Fatalf("lock holder: served by %s", got)
	}
}
