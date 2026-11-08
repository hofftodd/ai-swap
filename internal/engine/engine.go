// Package engine turns a model config into a runnable server command for a
// specific inference engine (Sushi, ds4, Strata, llama.cpp, or anything that
// speaks the OpenAI API).
package engine

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"

	"github.com/hofftodd/ai-swap/internal/config"
)

// Caps describes which client APIs an engine serves natively.
type Caps struct {
	OpenAI    bool `json:"openai"`
	Anthropic bool `json:"anthropic"`
}

// Launch is everything needed to start and probe one server process.
type Launch struct {
	Argv       []string
	Dir        string
	Env        []string
	Port       int
	HealthPath string
	// WarmupPath, if set, is POSTed once the health check passes; the model
	// counts as ready only when it answers 2xx. Engines that start serving
	// before their weights are loaded (oMLX) use it so "ready" means ready.
	WarmupPath string
}

// Engine is an inference-engine adapter.
type Engine interface {
	Name() string
	// Supported reports whether this engine can run on the current OS.
	Supported() bool
	// Caps reports which APIs the engine serves.
	Caps() Caps
	// Build produces the launch spec. allocPort is called only when the
	// engine can be told which port to use.
	Build(m *config.Model, ec config.EngineConfig, allocPort func() (int, error)) (*Launch, error)
}

// builtin is a table-driven adapter; most engines differ only in flag names.
type builtin struct {
	name       string
	bin        string   // default binary
	subcmd     []string // e.g. ["serve"]
	goos       []string // empty = any
	caps       Caps
	modelFlag  string
	ctxFlag    string
	portFlag   string // empty = engine uses fixedPort
	hostFlag   string
	fixedPort  int
	healthPath string
	extra      func(m *config.Model, ec config.EngineConfig) []string
	// Optional hooks for engines that address models differently.
	modelValue func(model string) string    // value passed to modelFlag
	warmup     func(m *config.Model) string // Launch.WarmupPath
	upstream   func(m *config.Model) string // default upstream_model
}

func (b *builtin) Name() string { return b.name }
func (b *builtin) Caps() Caps   { return b.caps }
func (b *builtin) Supported() bool {
	return len(b.goos) == 0 || slices.Contains(b.goos, runtime.GOOS)
}

func (b *builtin) Build(m *config.Model, ec config.EngineConfig, allocPort func() (int, error)) (*Launch, error) {
	portFlag := firstNonEmpty(ec.PortFlag, b.portFlag)
	hostFlag := firstNonEmpty(ec.HostFlag, b.hostFlag)

	l := &Launch{
		Dir:        ec.Dir,
		HealthPath: firstNonEmpty(m.HealthPath, b.healthPath, "/v1/models"),
		Env:        envList(ec.Env, m.Env),
	}
	if b.warmup != nil {
		l.WarmupPath = b.warmup(m)
	}

	switch {
	case m.Port != 0:
		l.Port = m.Port
	case portFlag != "" || strings.Contains(m.Cmd, "${PORT}"):
		p, err := allocPort()
		if err != nil {
			return nil, err
		}
		l.Port = p
	case b.fixedPort != 0:
		l.Port = b.fixedPort
	default:
		return nil, fmt.Errorf("model %s: engine %s has no port flag; set port, use ${PORT} in cmd, or set engines.%s.port_flag",
			m.Name, b.name, b.name)
	}

	if m.Cmd != "" {
		argv, err := SplitCommand(substitute(m.Cmd, m, l.Port))
		if err != nil {
			return nil, fmt.Errorf("model %s: cmd: %w", m.Name, err)
		}
		l.Argv = argv
		return l, nil
	}

	if b.bin == "" && ec.Bin == "" {
		return nil, fmt.Errorf("engine %s needs cmd", b.name)
	}
	argv := []string{firstNonEmpty(ec.Bin, b.bin)}
	argv = append(argv, b.subcmd...)
	if m.Model != "" && b.modelFlag != "" {
		v := m.Model
		if b.modelValue != nil {
			v = b.modelValue(v)
		}
		argv = append(argv, b.modelFlag, v)
	}
	if m.Ctx > 0 && b.ctxFlag != "" {
		argv = append(argv, b.ctxFlag, strconv.Itoa(m.Ctx))
	}
	if hostFlag != "" {
		argv = append(argv, hostFlag, "127.0.0.1")
	}
	if portFlag != "" {
		argv = append(argv, portFlag, strconv.Itoa(l.Port))
	}
	if b.extra != nil {
		argv = append(argv, b.extra(m, ec)...)
	}
	argv = append(argv, ec.Args...)
	argv = append(argv, m.Args...)
	l.Argv = argv
	return l, nil
}

var registry = map[string]Engine{}

func register(e Engine) { registry[e.Name()] = e }

// Get returns the adapter with the given name.
func Get(name string) (Engine, error) {
	e, ok := registry[name]
	if !ok {
		names := make([]string, 0, len(registry))
		for n := range registry {
			names = append(names, n)
		}
		slices.Sort(names)
		return nil, fmt.Errorf("unknown engine %q (known: %s)", name, strings.Join(names, ", "))
	}
	return e, nil
}

func init() {
	// llama.cpp's llama-server: the reference OpenAI-compatible server, and
	// the practical choice for the R9700 (ROCm or Vulkan builds).
	register(&builtin{
		name: "llamacpp", bin: "llama-server",
		caps:      Caps{OpenAI: true, Anthropic: true},
		modelFlag: "-m", ctxFlag: "-c", portFlag: "--port", hostFlag: "--host",
		healthPath: "/health",
		// --metrics enables the Prometheus endpoint the stats pane reads
		// (KV-cache use, token counters). Harmless if already given.
		extra: func(m *config.Model, ec config.EngineConfig) []string {
			if slices.Contains(m.Args, "--metrics") || slices.Contains(ec.Args, "--metrics") {
				return nil
			}
			return []string{"--metrics"}
		},
	})

	// Sushi: Apple Silicon only. `sushi --serve --model DIR --ctx-size N
	// --host H --port P` (checked against sushi 1.4.1 on an M5 Ultra).
	register(&builtin{
		name: "sushi", bin: "sushi", subcmd: []string{"--serve"},
		goos:      []string{"darwin"},
		caps:      Caps{OpenAI: true, Anthropic: true},
		modelFlag: "--model", ctxFlag: "--ctx-size", portFlag: "--port", hostFlag: "--host",
	})

	// ds4 (DwarfStar4): DeepSeek V4 Flash. `ds4-server -m FILE --ctx N
	// --host H --port P`, serves /v1/chat/completions and /v1/messages.
	// Builds exist for Metal, CUDA (DGX Spark) and ROCm (Strix Halo). It has
	// no working-directory flag; engines.ds4.dir sets the process's cwd.
	register(&builtin{
		name: "ds4", bin: "ds4-server",
		goos:      []string{"darwin", "linux"},
		caps:      Caps{OpenAI: true, Anthropic: true},
		modelFlag: "-m", ctxFlag: "--ctx", portFlag: "--port", hostFlag: "--host",
	})

	// oMLX: MLX models with continuous batching (Apple Silicon). It serves
	// every model directory under --model-dir and loads one on first use, so
	// `model:` is the model's own directory: the adapter passes its parent as
	// --model-dir, sends requests under the directory's name, and loads the
	// weights before reporting ready. Each model gets its own SSD prompt cache.
	register(&builtin{
		name: "omlx", bin: "omlx", subcmd: []string{"serve"},
		goos:      []string{"darwin"},
		caps:      Caps{OpenAI: true, Anthropic: true},
		modelFlag: "--model-dir", portFlag: "--port", hostFlag: "--host",
		modelValue: filepath.Dir,
		upstream:   func(m *config.Model) string { return filepath.Base(m.Model) },
		warmup: func(m *config.Model) string {
			if m.Model == "" {
				return ""
			}
			return "/v1/models/" + url.PathEscape(upstreamOf(m)) + "/load"
		},
		extra: func(m *config.Model, ec config.EngineConfig) []string {
			out := []string{"--headless"}
			if m.Model != "" && !slices.Contains(m.Args, "--paged-ssd-cache-dir") &&
				!slices.Contains(ec.Args, "--paged-ssd-cache-dir") {
				if home, err := os.UserHomeDir(); err == nil {
					out = append(out, "--paged-ssd-cache-dir",
						filepath.Join(home, ".omlx", "cache", "ai-swap", filepath.Base(m.Model)))
				}
			}
			return out
		},
	})

	// Strata: its documented entry point (`strata launch`) is interactive,
	// so a non-interactive server command must be given via cmd until the
	// adapter learns the real flags. Default address 127.0.0.1:8080.
	register(&builtin{
		name:      "strata",
		caps:      Caps{OpenAI: true, Anthropic: true},
		fixedPort: 8080,
	})

	// generic: any OpenAI-compatible server, fully described by cmd.
	register(&builtin{
		name: "generic",
		caps: Caps{OpenAI: true},
	})
}

// upstreamOf is the model name an oMLX-style engine expects.
func upstreamOf(m *config.Model) string {
	if m.UpstreamModel != "" {
		return m.UpstreamModel
	}
	return filepath.Base(m.Model)
}

// ApplyDefaults fills in engine-specific defaults that live on the model
// config (currently upstream_model for engines that address models by
// directory name). Call it once after loading the config.
func ApplyDefaults(cfg *config.Config) {
	for _, m := range cfg.Models {
		b, ok := registry[m.Engine].(*builtin)
		if !ok || b.upstream == nil || m.UpstreamModel != "" || m.Model == "" {
			continue
		}
		m.UpstreamModel = b.upstream(m)
	}
}

func substitute(s string, m *config.Model, port int) string {
	return strings.NewReplacer(
		"${PORT}", strconv.Itoa(port),
		"${MODEL}", m.Model,
		"${CTX}", strconv.Itoa(m.Ctx),
	).Replace(s)
}

func envList(maps ...map[string]string) []string {
	var out []string
	for _, mp := range maps {
		for k, v := range mp {
			out = append(out, k+"="+v)
		}
	}
	slices.Sort(out)
	return out
}

func firstNonEmpty(s ...string) string {
	for _, v := range s {
		if v != "" {
			return v
		}
	}
	return ""
}

// SplitCommand splits a command line into argv, honouring single quotes,
// double quotes and backslash escapes. It does not invoke a shell.
func SplitCommand(s string) ([]string, error) {
	var (
		args    []string
		cur     strings.Builder
		inArg   bool
		quote   rune
		escaped bool
	)
	for _, r := range s {
		switch {
		case escaped:
			cur.WriteRune(r)
			escaped = false
		case r == '\\' && quote != '\'':
			escaped, inArg = true, true
		case quote != 0:
			if r == quote {
				quote = 0
			} else {
				cur.WriteRune(r)
			}
		case r == '\'' || r == '"':
			quote, inArg = r, true
		case r == ' ' || r == '\t' || r == '\n':
			if inArg {
				args = append(args, cur.String())
				cur.Reset()
				inArg = false
			}
		default:
			cur.WriteRune(r)
			inArg = true
		}
	}
	if quote != 0 {
		return nil, fmt.Errorf("unterminated quote")
	}
	if escaped {
		return nil, fmt.Errorf("trailing backslash")
	}
	if inArg {
		args = append(args, cur.String())
	}
	if len(args) == 0 {
		return nil, fmt.Errorf("empty command")
	}
	return args, nil
}
