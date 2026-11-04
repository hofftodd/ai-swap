package server

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hofftodd/ai-swap/internal/cluster"
	"github.com/hofftodd/ai-swap/internal/config"
	"github.com/hofftodd/ai-swap/internal/engine"
	"github.com/hofftodd/ai-swap/internal/process"
	"github.com/hofftodd/ai-swap/internal/scheduler"
)

// backend is a fake inference server: it echoes the model and path it saw
// and streams three SSE chunks when asked to.
type backend struct {
	srv  *http.Server
	done chan struct{}
	once sync.Once
}

func (b *backend) WaitHealthy(context.Context, time.Duration) error { return nil }
func (b *backend) Done() <-chan struct{}                            { return b.done }
func (b *backend) Stop(time.Duration) {
	b.once.Do(func() { b.srv.Close(); close(b.done) })
}

func startBackend(l *engine.Launch, _ *process.RingBuffer) (scheduler.Handle, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(l.Port))
	if err != nil {
		return nil, err
	}
	name := l.Argv[0]
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for i := 0; i < 3; i++ {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"%d\"}}]}\n\n", i)
				w.(http.Flusher).Flush()
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		writeJSON(w, 200, map[string]string{"backend": name, "model": req.Model, "path": r.URL.Path})
	})
	b := &backend{srv: &http.Server{Handler: mux}, done: make(chan struct{})}
	go b.srv.Serve(ln)
	return b, nil
}

type node struct {
	url string
	mgr *scheduler.Manager
}

func newNode(t *testing.T, yaml string, peers ...string) *node {
	t.Helper()
	cfg, err := config.Parse([]byte(yaml))
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range peers {
		cfg.Peers = append(cfg.Peers, config.Peer{Name: fmt.Sprintf("peer%d", i), URL: p})
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr, err := scheduler.New(cfg, log)
	if err != nil {
		t.Fatal(err)
	}
	mgr.SetStartFunc(startBackend)
	cl, err := cluster.New(cfg.Peers, log)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go cl.Run(ctx, time.Hour)
	ts := httptest.NewServer(New(cfg, mgr, cl, nil, log).Handler())
	t.Cleanup(func() { cancel(); ts.Close(); mgr.StopAll() })
	if len(peers) > 0 {
		deadline := time.Now().Add(2 * time.Second)
		for !allOnline(cl) {
			if time.Now().After(deadline) {
				t.Fatal("peers never came online")
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	return &node{url: ts.URL, mgr: mgr}
}

func allOnline(cl *cluster.Cluster) bool {
	for _, p := range cl.Peers() {
		if !p.Online {
			return false
		}
	}
	return true
}

func post(t *testing.T, url, body string) (int, map[string]any) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

const linuxCfg = `
node: {name: linux, port_range: [19300, 19349]}
models:
  coder:
    engine: generic
    cmd: "coder-backend --port ${PORT}"
    aliases: [c]
`

const macCfg = `
node: {name: mac, port_range: [19350, 19399]}
models:
  qwen:
    engine: generic
    cmd: "qwen-backend --port ${PORT}"
    upstream_model: Qwen3.8-Flash-Next
`

func TestLocalRouting(t *testing.T) {
	mac := newNode(t, macCfg)
	code, out := post(t, mac.url+"/v1/chat/completions", `{"model":"qwen","messages":[]}`)
	if code != 200 {
		t.Fatalf("status %d %v", code, out)
	}
	if out["backend"] != "qwen-backend" || out["model"] != "Qwen3.8-Flash-Next" || out["path"] != "/v1/chat/completions" {
		t.Fatalf("unexpected upstream view: %v", out)
	}
}

func TestPeerRouting(t *testing.T) {
	linux := newNode(t, linuxCfg)
	mac := newNode(t, macCfg, linux.url)

	// Alias on the peer is rewritten to the peer's real model name.
	code, out := post(t, mac.url+"/v1/messages", `{"model":"c","max_tokens":5,"messages":[]}`)
	if code != 200 || out["backend"] != "coder-backend" || out["model"] != "coder" || out["path"] != "/v1/messages" {
		t.Fatalf("forward failed: %d %v", code, out)
	}

	// model@node pins a node.
	code, out = post(t, mac.url+"/v1/chat/completions", `{"model":"coder@linux"}`)
	if code != 200 || out["model"] != "coder" {
		t.Fatalf("pinned forward failed: %d %v", code, out)
	}

	// /v1/models on mac lists both nodes' models.
	resp, err := http.Get(mac.url + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	var list struct {
		Data []modelEntry `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&list)
	resp.Body.Close()
	got := map[string]string{}
	for _, m := range list.Data {
		got[m.ID] = m.Node
	}
	if got["qwen"] != "mac" || got["coder"] != "peer0" {
		t.Fatalf("models list: %v", got)
	}
}

func TestUnknownModelErrorShapes(t *testing.T) {
	mac := newNode(t, macCfg)
	code, out := post(t, mac.url+"/v1/messages", `{"model":"nope"}`)
	if code != 404 || out["type"] != "error" {
		t.Fatalf("anthropic-style error expected: %d %v", code, out)
	}
	code, out = post(t, mac.url+"/v1/chat/completions", `{"model":"nope"}`)
	if code != 404 || out["error"] == nil || out["type"] != nil {
		t.Fatalf("openai-style error expected: %d %v", code, out)
	}
	code, _ = post(t, mac.url+"/v1/chat/completions", `{}`)
	if code != 400 {
		t.Fatalf("missing model should be 400, got %d", code)
	}
}

func TestStreamingPassesThrough(t *testing.T) {
	linux := newNode(t, linuxCfg)
	mac := newNode(t, macCfg, linux.url)
	resp, err := http.Post(mac.url+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"coder","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type %q", ct)
	}
	var chunks []string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if line := sc.Text(); strings.HasPrefix(line, "data: ") {
			chunks = append(chunks, line)
		}
	}
	if len(chunks) != 4 || chunks[3] != "data: [DONE]" {
		t.Fatalf("chunks: %q", chunks)
	}
}

func TestLoadUnloadAPI(t *testing.T) {
	linux := newNode(t, linuxCfg)
	mac := newNode(t, macCfg, linux.url)
	for _, step := range []struct{ path, want string }{
		{"/api/models/coder@linux/load", "ready"},
		{"/api/models/coder@linux/unload", "stopped"},
	} {
		resp, err := http.Post(mac.url+step.path, "", nil)
		if err != nil || resp.StatusCode != 200 {
			t.Fatalf("%s: %v %v", step.path, err, resp.Status)
		}
		resp.Body.Close()
		if s := linux.mgr.Snapshot()[0].State; string(s) != step.want {
			t.Fatalf("%s: state %s, want %s", step.path, s, step.want)
		}
	}
}
