package config

import (
	"os"
	"strings"
	"testing"
	"time"
)

func TestParseDefaults(t *testing.T) {
	c, err := Parse([]byte(`
node:
  name: mac
  memory_budget_gb: 100
defaults:
  ttl: 30m
models:
  a:
    engine: llamacpp
    model: /m/a.gguf
    memory_gb: 40
    aliases: [alpha]
  b:
    engine: sushi
    memory_gb: 60
    ttl: 0
`))
	if err != nil {
		t.Fatal(err)
	}
	if c.Node.Listen != "127.0.0.1:8686" {
		t.Errorf("listen = %q", c.Node.Listen)
	}
	if c.Models["a"].TTL.D() != 30*time.Minute {
		t.Errorf("a ttl = %v", c.Models["a"].TTL.D())
	}
	if c.Models["b"].TTL.D() != 0 {
		t.Errorf("b ttl = %v, want 0 (explicit)", c.Models["b"].TTL.D())
	}
	if c.Models["a"].HealthTimeout.D() != 10*time.Minute {
		t.Errorf("health timeout = %v", c.Models["a"].HealthTimeout.D())
	}
	if m, ok := c.Resolve("alpha"); !ok || m.Name != "a" {
		t.Errorf("alias did not resolve: %v %v", m, ok)
	}
}

func TestValidate(t *testing.T) {
	_, err := Parse([]byte(`
node: {memory_budget_gb: 32}
models:
  big: {engine: llamacpp, memory_gb: 64}
  g: {engine: generic}
  x@y: {engine: llamacpp}
  dup: {engine: llamacpp, aliases: [big]}
  noengine: {}
`))
	if err == nil {
		t.Fatal("expected errors")
	}
	for _, want := range []string{"exceeds node budget", "generic engine needs cmd", "'@'", `name "big" used by both`, "engine is required"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error missing %q:\n%v", want, err)
		}
	}
}

func TestUnknownFieldRejected(t *testing.T) {
	if _, err := Parse([]byte("models:\n  a: {engine: llamacpp, memroy_gb: 4}\n")); err == nil {
		t.Fatal("typo in field name should be rejected")
	}
}

func TestExampleConfigsParse(t *testing.T) {
	for _, f := range []string{"../../examples/mac.yaml", "../../examples/linux-r9700.yaml"} {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Parse(b); err != nil {
			t.Errorf("%s: %v", f, err)
		}
	}
}
