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
			model:    config.Model{Name: "s", Model: "/m/sushi", Ctx: 131072},
			wantArgv: []string{"sushi", "serve", "--model", "/m/sushi", "--ctx-size", "131072"},
			wantPort: 12345,
		},
		{
			engine:   "sushi",
			model:    config.Model{Name: "s", Model: "/m/sushi"},
			ec:       config.EngineConfig{Bin: "/opt/homebrew/bin/sushi", PortFlag: "--port"},
			wantArgv: []string{"/opt/homebrew/bin/sushi", "serve", "--model", "/m/sushi", "--port", "9100"},
			wantPort: 9100,
		},
		{
			engine:   "ds4",
			model:    config.Model{Name: "d", Model: "ds4flash.gguf", Ctx: 32768},
			ec:       config.EngineConfig{Bin: "/src/ds4/ds4-server", Dir: "/src/ds4", Args: []string{"--kv-disk-dir", "/tmp/kv"}},
			wantArgv: []string{"/src/ds4/ds4-server", "-m", "ds4flash.gguf", "--ctx", "32768", "--chdir", "/src/ds4", "--kv-disk-dir", "/tmp/kv"},
			wantPort: 8000,
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
