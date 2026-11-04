// Package config loads and validates the ai-swap YAML configuration.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the top-level configuration for one ai-swap node.
type Config struct {
	Node     Node                    `yaml:"node"`
	Peers    []Peer                  `yaml:"peers"`
	Defaults Defaults                `yaml:"defaults"`
	Engines  map[string]EngineConfig `yaml:"engines"`
	Models   map[string]*Model       `yaml:"models"`
}

// Node describes the machine this ai-swap instance runs on.
type Node struct {
	// Name identifies this node to peers and in the UI. Defaults to the hostname.
	Name string `yaml:"name"`
	// Listen is the address the proxy and UI listen on.
	Listen string `yaml:"listen"`
	// MemoryBudgetGB is how much (V)RAM models on this node may use at once.
	// On Apple Silicon / Strix Halo / DGX Spark this is a share of unified
	// memory; on a discrete GPU it is the card's VRAM. 0 disables the check.
	MemoryBudgetGB float64 `yaml:"memory_budget_gb"`
	// PortRange is the range of ports handed to engines that accept a port flag.
	PortRange [2]int `yaml:"port_range"`
}

// Peer is another ai-swap node whose models this node can route to.
type Peer struct {
	Name string `yaml:"name"`
	URL  string `yaml:"url"`
}

// Defaults apply to every model unless the model overrides them.
type Defaults struct {
	TTL           Duration `yaml:"ttl"`
	HealthTimeout Duration `yaml:"health_timeout"`
}

// EngineConfig holds per-node install details for an engine.
type EngineConfig struct {
	// Bin is the server binary. Defaults to the engine's usual binary name.
	Bin string `yaml:"bin"`
	// Dir is the working directory for the server process.
	Dir string `yaml:"dir"`
	// PortFlag / HostFlag override the engine's flags for port and host.
	// Setting PortFlag lets an engine that otherwise uses a fixed port run
	// on a port from the node's port range.
	PortFlag string `yaml:"port_flag"`
	HostFlag string `yaml:"host_flag"`
	// Args are appended to every model of this engine.
	Args []string `yaml:"args"`
	// Env is added to the environment of every model of this engine.
	Env map[string]string `yaml:"env"`
}

// Model is one servable model.
type Model struct {
	// Name is filled in from the map key.
	Name string `yaml:"-"`
	// Engine selects the adapter: sushi, ds4, strata, llamacpp, generic.
	Engine string `yaml:"engine"`
	// Model is the model path or identifier the engine expects.
	Model string `yaml:"model"`
	// Ctx is the context length; 0 leaves it to the engine.
	Ctx int `yaml:"ctx"`
	// Args are extra command-line arguments.
	Args []string `yaml:"args"`
	// Env is extra environment for the process.
	Env map[string]string `yaml:"env"`
	// Cmd fully replaces the generated command line. ${PORT}, ${MODEL} and
	// ${CTX} are substituted. Required for the generic engine.
	Cmd string `yaml:"cmd"`
	// Port forces a fixed upstream port.
	Port int `yaml:"port"`
	// MemoryGB is the expected footprint while loaded, used for scheduling.
	MemoryGB float64 `yaml:"memory_gb"`
	// TTL unloads the model after this much idle time. 0 means never.
	TTL *Duration `yaml:"ttl"`
	// HealthTimeout is how long to wait for the server to become ready.
	HealthTimeout *Duration `yaml:"health_timeout"`
	// HealthPath overrides the engine's readiness endpoint.
	HealthPath string `yaml:"health_path"`
	// Aliases are extra names clients may use for this model.
	Aliases []string `yaml:"aliases"`
	// UpstreamModel, if set, replaces the "model" field sent to the backend.
	UpstreamModel string `yaml:"upstream_model"`
}

// Duration is a time.Duration that unmarshals from strings like "30m".
type Duration time.Duration

// UnmarshalYAML implements yaml.Unmarshaler.
func (d *Duration) UnmarshalYAML(n *yaml.Node) error {
	var s string
	if err := n.Decode(&s); err != nil {
		return err
	}
	if s == "" || s == "0" {
		*d = 0
		return nil
	}
	v, err := time.ParseDuration(s)
	if err != nil {
		return fmt.Errorf("invalid duration %q: %w", s, err)
	}
	*d = Duration(v)
	return nil
}

// D returns the value as a time.Duration.
func (d Duration) D() time.Duration { return time.Duration(d) }

// Load reads, defaults and validates a config file.
func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(b)
}

// Parse decodes, defaults and validates config bytes.
func Parse(b []byte) (*Config, error) {
	var c Config
	dec := yaml.NewDecoder(strings.NewReader(string(b)))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	c.applyDefaults()
	if err := c.validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

func (c *Config) applyDefaults() {
	if c.Node.Name == "" {
		c.Node.Name, _ = os.Hostname()
	}
	if c.Node.Listen == "" {
		c.Node.Listen = "127.0.0.1:8686"
	}
	if c.Node.PortRange == [2]int{} {
		c.Node.PortRange = [2]int{9100, 9199}
	}
	if c.Defaults.HealthTimeout == 0 {
		// Large MoE models can take minutes to map into memory.
		c.Defaults.HealthTimeout = Duration(10 * time.Minute)
	}
	if c.Engines == nil {
		c.Engines = map[string]EngineConfig{}
	}
	for name, m := range c.Models {
		if m == nil {
			m = &Model{}
			c.Models[name] = m
		}
		m.Name = name
		m.Model = ExpandHome(m.Model)
		m.Args = expandAll(m.Args)
		if m.TTL == nil {
			t := c.Defaults.TTL
			m.TTL = &t
		}
		if m.HealthTimeout == nil {
			t := c.Defaults.HealthTimeout
			m.HealthTimeout = &t
		}
	}
	for name, e := range c.Engines {
		e.Bin = ExpandHome(e.Bin)
		e.Dir = ExpandHome(e.Dir)
		e.Args = expandAll(e.Args)
		c.Engines[name] = e
	}
}

func (c *Config) validate() error {
	var errs []string
	if c.Node.PortRange[0] <= 0 || c.Node.PortRange[1] < c.Node.PortRange[0] {
		errs = append(errs, fmt.Sprintf("node.port_range %v is invalid", c.Node.PortRange))
	}
	seen := map[string]string{}
	for _, name := range c.ModelNames() {
		m := c.Models[name]
		if m.Engine == "" {
			errs = append(errs, fmt.Sprintf("model %q: engine is required", name))
		}
		if m.Engine == "generic" && m.Cmd == "" {
			errs = append(errs, fmt.Sprintf("model %q: generic engine needs cmd", name))
		}
		if m.MemoryGB < 0 {
			errs = append(errs, fmt.Sprintf("model %q: memory_gb must be >= 0", name))
		}
		if c.Node.MemoryBudgetGB > 0 && m.MemoryGB > c.Node.MemoryBudgetGB {
			errs = append(errs, fmt.Sprintf("model %q: memory_gb %.0f exceeds node budget %.0f",
				name, m.MemoryGB, c.Node.MemoryBudgetGB))
		}
		if strings.Contains(name, "@") {
			errs = append(errs, fmt.Sprintf("model %q: names may not contain '@' (reserved for model@node)", name))
		}
		for _, n := range append([]string{name}, m.Aliases...) {
			if other, ok := seen[n]; ok {
				errs = append(errs, fmt.Sprintf("name %q used by both %q and %q", n, other, name))
			}
			seen[n] = name
		}
	}
	for _, p := range c.Peers {
		if p.Name == "" || p.URL == "" {
			errs = append(errs, "peers need both name and url")
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("invalid config:\n  %s", strings.Join(errs, "\n  "))
	}
	return nil
}

// ModelNames returns model names in sorted order.
func (c *Config) ModelNames() []string {
	names := make([]string, 0, len(c.Models))
	for n := range c.Models {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Resolve maps a requested name or alias to a local model.
func (c *Config) Resolve(name string) (*Model, bool) {
	if m, ok := c.Models[name]; ok {
		return m, true
	}
	for _, m := range c.Models {
		for _, a := range m.Aliases {
			if a == name {
				return m, true
			}
		}
	}
	return nil, false
}

func expandAll(args []string) []string {
	for i, a := range args {
		args[i] = ExpandHome(a)
	}
	return args
}

// ExpandHome expands a leading ~/ to the user's home directory.
func ExpandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	return p
}
