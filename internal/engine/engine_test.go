package engine

import (
	"reflect"
	"strings"
	"testing"

	"github.com/hofftodd/ai-swap/internal/config"
)

func alloc(p int) func() (int, error) { return func() (int, error) { return p, nil } }

func TestBuildCommands(t *testing.T) {
	tests := []struct {
		engine   string
		model    config.Model
		ec       config.EngineConfig
		wantArgv []string
		wantPort int
	}{
		{
			engine:   "llamacpp",
			model:    config.Model{Name: "q", Model: "/m/q.gguf", Ctx: 8192, Args: []string{"-ngl", "99"}},
			wantArgv: []string{"llama-server", "-m", "/m/q.gguf", "-c", "8192", "--host", "127.0.0.1", "--port", "9100", "--metrics", "-ngl", "99"},
			wantPort: 9100,
		},
		{
			engine:   "sushi",
			model:    config.Model{Name: "s", Model: "/m/sushi", Ctx: 131072, Args: []string{"--no-vision"}},
			ec:       config.EngineConfig{Bin: "/opt/homebrew/bin/sushi"},
			wantArgv: []string{"/opt/homebrew/bin/sushi", "--serve", "--model", "/m/sushi", "--ctx-size", "131072", "--host", "127.0.0.1", "--port", "9100", "--no-vision"},
			wantPort: 9100,
		},
		{
			engine:   "ds4",
			model:    config.Model{Name: "d", Model: "ds4flash.gguf", Ctx: 32768},
			ec:       config.EngineConfig{Bin: "/src/ds4/ds4-server", Dir: "/src/ds4", Args: []string{"--kv-disk-dir", "/tmp/kv"}},
			wantArgv: []string{"/src/ds4/ds4-server", "-m", "ds4flash.gguf", "--ctx", "32768", "--host", "127.0.0.1", "--port", "9100", "--kv-disk-dir", "/tmp/kv"},
			wantPort: 9100,
		},
		{
			engine:   "omlx",
			model:    config.Model{Name: "f", Model: "/m/omlx/flint-oq4e", Args: []string{"--paged-ssd-cache-dir", "/c"}},
			wantArgv: []string{"omlx", "serve", "--model-dir", "/m/omlx", "--host", "127.0.0.1", "--port", "9100", "--headless", "--paged-ssd-cache-dir", "/c"},
			wantPort: 9100,
		},
		{
			engine:   "strata",
			model:    config.Model{Name: "st", Model: "qwen", Cmd: `strata serve --model ${MODEL} --port ${PORT} --name "my model"`},
			wantArgv: []string{"strata", "serve", "--model", "qwen", "--port", "9100", "--name", "my model"},
			wantPort: 9100,
		},
		{
			engine:   "strata",
			model:    config.Model{Name: "st", Cmd: "strata-server"},
			wantArgv: []string{"strata-server"},
			wantPort: 8080,
		},
	}
	for _, tt := range tests {
		t.Run(tt.engine, func(t *testing.T) {
			e, err := Get(tt.engine)
			if err != nil {
				t.Fatal(err)
			}
			l, err := e.Build(&tt.model, tt.ec, alloc(9100))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(l.Argv, tt.wantArgv) {
				t.Errorf("argv\n got %q\nwant %q", l.Argv, tt.wantArgv)
			}
			if l.Port != tt.wantPort {
				t.Errorf("port = %d, want %d", l.Port, tt.wantPort)
			}
		})
	}
}

func TestOMLXWarmupAndUpstream(t *testing.T) {
	cfg := &config.Config{Models: map[string]*config.Model{
		"flint":  {Name: "flint", Engine: "omlx", Model: "/m/omlx/flint-oq4e"},
		"pinned": {Name: "pinned", Engine: "omlx", Model: "/m/omlx/x", UpstreamModel: "custom id"},
		"llama":  {Name: "llama", Engine: "llamacpp", Model: "/m/q.gguf"},
	}}
	ApplyDefaults(cfg)
	if got := cfg.Models["flint"].UpstreamModel; got != "flint-oq4e" {
		t.Errorf("flint upstream = %q", got)
	}
	if got := cfg.Models["pinned"].UpstreamModel; got != "custom id" {
		t.Errorf("pinned upstream overwritten: %q", got)
	}
	if got := cfg.Models["llama"].UpstreamModel; got != "" {
		t.Errorf("llamacpp got an upstream: %q", got)
	}
	e, _ := Get("omlx")
	for name, want := range map[string]string{"flint": "/v1/models/flint-oq4e/load", "pinned": "/v1/models/custom%20id/load"} {
		l, err := e.Build(cfg.Models[name], config.EngineConfig{}, alloc(9100))
		if err != nil {
			t.Fatal(err)
		}
		if l.WarmupPath != want {
			t.Errorf("%s warmup = %q, want %q", name, l.WarmupPath, want)
		}
	}
	if l, _ := Get("llamacpp"); l != nil {
		ll, _ := l.Build(cfg.Models["llama"], config.EngineConfig{}, alloc(9100))
		if ll.WarmupPath != "" {
			t.Errorf("llamacpp warmup = %q", ll.WarmupPath)
		}
	}
}

func TestGenericNeedsPort(t *testing.T) {
	e, _ := Get("generic")
	_, err := e.Build(&config.Model{Name: "g", Cmd: "server"}, config.EngineConfig{}, alloc(9100))
	if err == nil || !strings.Contains(err.Error(), "port") {
		t.Fatalf("want port error, got %v", err)
	}
	l, err := e.Build(&config.Model{Name: "g", Cmd: "server", Port: 7000}, config.EngineConfig{}, alloc(9100))
	if err != nil || l.Port != 7000 {
		t.Fatalf("fixed port: %v %+v", err, l)
	}
}

func TestSplitCommand(t *testing.T) {
	got, err := SplitCommand(`a 'b c' "d \"e\"" f\ g   `)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"a", "b c", `d "e"`, "f g"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
	for _, bad := range []string{"", "  ", `a "b`, `a\`} {
		if _, err := SplitCommand(bad); err == nil {
			t.Errorf("SplitCommand(%q) should fail", bad)
		}
	}
}

func TestUnknownEngine(t *testing.T) {
	if _, err := Get("nope"); err == nil || !strings.Contains(err.Error(), "sushi") {
		t.Fatalf("want error listing engines, got %v", err)
	}
}

func TestSplash(t *testing.T) {
	cfg := &config.Config{Models: map[string]*config.Model{
		"flint-splash": {Name: "flint-splash", Engine: "splash", Model: "/m/flint/flint-UD-Q4_K_XL.gguf", Ctx: 131072},
		"hub":          {Name: "hub", Engine: "splash", Model: "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M", Args: []string{"--idle-release", "30m"}},
		"pinned":       {Name: "pinned", Engine: "splash", Model: "/m/x", UpstreamModel: "me/x"},
	}}
	ApplyDefaults(cfg)
	e, err := Get("splash")
	if err != nil {
		t.Fatal(err)
	}
	if !e.Caps().Anthropic || !e.Caps().OpenAI {
		t.Errorf("caps = %+v", e.Caps())
	}
	tests := []struct {
		name, upstream string
		argv           []string
	}{
		{"flint-splash", "ai-swap/flint-splash", []string{"splash", "serve", "--max-context", "131072", "--host", "127.0.0.1", "--port", "9100",
			"--model", "ai-swap/flint-splash", "--model-dir", "/m/flint/flint-UD-Q4_K_XL.gguf", "--idle-release", "off", "--no-webui"}},
		{"hub", "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M", []string{"splash", "serve", "--host", "127.0.0.1", "--port", "9100",
			"--model", "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M", "--no-webui", "--idle-release", "30m"}},
		{"pinned", "me/x", []string{"splash", "serve", "--host", "127.0.0.1", "--port", "9100",
			"--model", "me/x", "--model-dir", "/m/x", "--idle-release", "off", "--no-webui"}},
	}
	for _, tt := range tests {
		m := cfg.Models[tt.name]
		if m.UpstreamModel != tt.upstream {
			t.Errorf("%s upstream = %q, want %q", tt.name, m.UpstreamModel, tt.upstream)
		}
		l, err := e.Build(m, config.EngineConfig{}, alloc(9100))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(l.Argv, tt.argv) {
			t.Errorf("%s argv\n got %q\nwant %q", tt.name, l.Argv, tt.argv)
		}
		if l.HealthPath != "/ready" || l.Port != 9100 {
			t.Errorf("%s health %q port %d", tt.name, l.HealthPath, l.Port)
		}
	}
}
