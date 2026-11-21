package config

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const stateCfg = `
node: {memory_budget_gb: 50, resident: [a]}
models:
  a: {engine: generic, cmd: "a ${PORT}", memory_gb: 20, aliases: [alpha]}
  b: {engine: generic, cmd: "b ${PORT}", memory_gb: 20}
  c: {engine: generic, cmd: "c ${PORT}", memory_gb: 40}
`

func TestStateRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), StateFileName)
	st, err := LoadState(path)
	if err != nil || !st.IsZero() {
		t.Fatalf("missing file: %+v %v", st, err)
	}
	res, def := []string{"b"}, "b"
	if err := SaveState(path, State{Resident: &res, DefaultModel: &def}); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(path)
	if !strings.HasPrefix(string(b), "# Written by ai-swap") {
		t.Fatalf("no header:\n%s", b)
	}
	st, err = LoadState(path)
	if err != nil || st.Resident == nil || !reflect.DeepEqual(*st.Resident, res) || *st.DefaultModel != "b" {
		t.Fatalf("round trip: %+v %v", st, err)
	}

	// An explicitly empty resident list survives (it means "none", not "unset").
	none := []string{}
	if err := SaveState(path, State{Resident: &none}); err != nil {
		t.Fatal(err)
	}
	if st, _ = LoadState(path); st.Resident == nil || len(*st.Resident) != 0 || st.DefaultModel != nil {
		t.Fatalf("empty list: %+v", st)
	}

	// Nothing left to override removes the file.
	if err := SaveState(path, State{}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("empty state should remove the file: %v", err)
	}
	if err := os.WriteFile(path, []byte("bogus: 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadState(path); err == nil {
		t.Fatal("unknown keys should be rejected")
	}
	// Empty, comments only, or "{}" (what yq leaves after llama-config deletes
	// the last key): nothing overridden.
	for _, body := range []string{"", stateHeader, stateHeader + "{}\n"} {
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if st, err := LoadState(path); err != nil || !st.IsZero() {
			t.Fatalf("%q: %+v %v", body, st, err)
		}
	}
}

func TestApplyState(t *testing.T) {
	c, err := Parse([]byte(stateCfg))
	if err != nil {
		t.Fatal(err)
	}
	res, def := []string{"alpha", "b"}, "alpha"
	if w := c.ApplyState(State{Resident: &res, DefaultModel: &def}); len(w) != 0 {
		t.Fatalf("warnings: %v", w)
	}
	if !reflect.DeepEqual(c.Node.Resident, []string{"a", "b"}) || c.Node.DefaultModel != "a" {
		t.Fatalf("not applied (aliases canonicalised): %v %q", c.Node.Resident, c.Node.DefaultModel)
	}

	// Settings that no longer fit the config (e.g. after llama-config
	// regenerated it without a model) are skipped with a warning.
	gone, over, peer := []string{"gone"}, []string{"b", "c"}, "x@studio"
	w := c.ApplyState(State{Resident: &gone})
	w = append(w, c.ApplyState(State{Resident: &over})...)
	missing := "gone"
	w = append(w, c.ApplyState(State{DefaultModel: &missing})...)
	if len(w) != 3 || !reflect.DeepEqual(c.Node.Resident, []string{"a", "b"}) || c.Node.DefaultModel != "a" {
		t.Fatalf("bad settings applied: %v %v %q", w, c.Node.Resident, c.Node.DefaultModel)
	}
	if w := c.ApplyState(State{DefaultModel: &peer}); len(w) != 0 || c.Node.DefaultModel != peer {
		t.Fatalf("peer default: %v %q", w, c.Node.DefaultModel)
	}
}

func TestDefaultModelConfig(t *testing.T) {
	c, err := Parse([]byte(strings.Replace(stateCfg, "resident: [a]", "resident: [a], default_model: alpha", 1)))
	if err != nil || c.Node.DefaultModel != "a" {
		t.Fatalf("default_model alias: %v %q", err, c.Node.DefaultModel)
	}
	if _, err := Parse([]byte(strings.Replace(stateCfg, "resident: [a]", "default_model: nope", 1))); err == nil ||
		!strings.Contains(err.Error(), "node.default_model") {
		t.Fatalf("unknown default_model accepted: %v", err)
	}
}
