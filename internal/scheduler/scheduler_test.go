package scheduler

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/hofftodd/ai-swap/internal/config"
	"github.com/hofftodd/ai-swap/internal/engine"
	"github.com/hofftodd/ai-swap/internal/process"
)

type fakeHandle struct {
	done    chan struct{}
	once    sync.Once
	healthy error
}

func (f *fakeHandle) WaitHealthy(context.Context, time.Duration) error { return f.healthy }
func (f *fakeHandle) Stop(time.Duration)                               { f.once.Do(func() { close(f.done) }) }
func (f *fakeHandle) Done() <-chan struct{}                            { return f.done }

type fakeRunner struct {
	mu      sync.Mutex
	started []string
	fail    map[string]error
}

func (r *fakeRunner) start(l *engine.Launch, _ *process.RingBuffer) (Handle, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	name := l.Argv[0]
	r.started = append(r.started, name)
	return &fakeHandle{done: make(chan struct{}), healthy: r.fail[name]}, nil
}

func newTestManager(t *testing.T, yaml string) (*Manager, *fakeRunner) {
	t.Helper()
	cfg, err := config.Parse([]byte(yaml))
	if err != nil {
		t.Fatal(err)
	}
	m, err := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	r := &fakeRunner{fail: map[string]error{}}
	m.SetStartFunc(r.start)
	return m, r
}

func states(m *Manager) map[string]State {
	out := map[string]State{}
	for _, s := range m.Snapshot() {
		out[s.Name] = s.State
	}
	return out
}

const twoBig = `
node: {memory_budget_gb: 100}
models:
  a: {engine: generic, cmd: "a --port ${PORT}", memory_gb: 60}
  b: {engine: generic, cmd: "b --port ${PORT}", memory_gb: 60}
  c: {engine: generic, cmd: "c --port ${PORT}", memory_gb: 30}
`

func TestSwapEvictsWhenOverBudget(t *testing.T) {
	m, r := newTestManager(t, twoBig)
	ctx := context.Background()

	_, rel, err := m.Acquire(ctx, "a")
	if err != nil {
		t.Fatal(err)
	}
	rel()
	_, rel, err = m.Acquire(ctx, "c") // 60+30 fits
	if err != nil {
		t.Fatal(err)
	}
	rel()
	if s := states(m); s["a"] != Ready || s["c"] != Ready {
		t.Fatalf("both should fit: %v", s)
	}

	// b (60) needs a to go; c is more recent so it stays.
	_, rel, err = m.Acquire(ctx, "b")
	if err != nil {
		t.Fatal(err)
	}
	rel()
	if s := states(m); s["a"] != Stopped || s["b"] != Ready || s["c"] != Ready {
		t.Fatalf("expected a evicted (LRU): %v", s)
	}
	if got := m.UsedGB(); got != 90 {
		t.Fatalf("used = %v, want 90", got)
	}
	if len(r.started) != 3 {
		t.Fatalf("starts = %v", r.started)
	}
}

func TestSwapWaitsForBusyModel(t *testing.T) {
	m, _ := newTestManager(t, twoBig)
	ctx := context.Background()

	_, relA, err := m.Acquire(ctx, "a")
	if err != nil {
		t.Fatal(err)
	}

	// With a busy, b cannot load; a short deadline must expire.
	short, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	if _, _, err := m.Acquire(short, "b"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want deadline exceeded, got %v", err)
	}

	got := make(chan error, 1)
	go func() {
		_, rel, err := m.Acquire(ctx, "b")
		if err == nil {
			rel()
		}
		got <- err
	}()
	time.Sleep(30 * time.Millisecond)
	if s := states(m); s["b"] != Stopped {
		t.Fatalf("b should still be waiting: %v", s)
	}
	relA()
	select {
	case err := <-got:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("b never loaded after a was released")
	}
	if s := states(m); s["a"] != Stopped || s["b"] != Ready {
		t.Fatalf("after swap: %v", s)
	}
}

func TestFixedPortConflict(t *testing.T) {
	m, _ := newTestManager(t, `
models:
  x: {engine: generic, cmd: "x", port: 7000, memory_gb: 1}
  y: {engine: generic, cmd: "y", port: 7000, memory_gb: 1}
`)
	ctx := context.Background()
	for _, n := range []string{"x", "y"} {
		_, rel, err := m.Acquire(ctx, n)
		if err != nil {
			t.Fatal(err)
		}
		rel()
	}
	if s := states(m); s["x"] != Stopped || s["y"] != Ready {
		t.Fatalf("same fixed port must not run twice: %v", s)
	}
}

func TestHealthFailure(t *testing.T) {
	m, r := newTestManager(t, twoBig)
	r.fail["a"] = errors.New("boom")
	if _, _, err := m.Acquire(context.Background(), "a"); err == nil {
		t.Fatal("expected error")
	}
	snap := m.Snapshot()
	if snap[0].State != Failed || snap[0].Error == "" {
		t.Fatalf("want failed with error, got %+v", snap[0])
	}
	if m.UsedGB() != 0 {
		t.Fatalf("failed model must not hold memory")
	}
}

func TestReaper(t *testing.T) {
	m, _ := newTestManager(t, `
defaults: {ttl: 1m}
models:
  a: {engine: generic, cmd: "a --port ${PORT}"}
  keep: {engine: generic, cmd: "k --port ${PORT}", ttl: 0}
`)
	ctx := context.Background()
	for _, n := range []string{"a", "keep"} {
		_, rel, _ := m.Acquire(ctx, n)
		rel()
	}
	m.reapOnce(time.Now().Add(30 * time.Second))
	if s := states(m); s["a"] != Ready {
		t.Fatalf("reaped too early: %v", s)
	}
	m.reapOnce(time.Now().Add(2 * time.Minute))
	if s := states(m); s["a"] != Stopped || s["keep"] != Ready {
		t.Fatalf("after ttl: %v", s)
	}
}

func TestCrashDetected(t *testing.T) {
	m, _ := newTestManager(t, twoBig)
	_, rel, _ := m.Acquire(context.Background(), "a")
	rel()
	inst, _ := m.Get("a")
	m.mu.Lock()
	h := inst.handle.(*fakeHandle)
	m.mu.Unlock()
	h.Stop(0) // simulate the process dying on its own
	deadline := time.Now().Add(time.Second)
	for states(m)["a"] != Failed {
		if time.Now().After(deadline) {
			t.Fatalf("crash not noticed: %v", states(m))
		}
		time.Sleep(5 * time.Millisecond)
	}
	// Next request restarts it.
	if _, rel, err := m.Acquire(context.Background(), "a"); err != nil {
		t.Fatal(err)
	} else {
		rel()
	}
}

// One-at-a-time card: r is resident, x is swapped in on demand.
const residentCfg = `
node: {memory_budget_gb: 100, resident: [r]}
defaults: {ttl: 1m}
models:
  r: {engine: generic, cmd: "r --port ${PORT}", memory_gb: 80}
  x: {engine: generic, cmd: "x --port ${PORT}", memory_gb: 80}
`

func TestResidentLifecycle(t *testing.T) {
	m, r := newTestManager(t, residentCfg)
	ctx := context.Background()

	// Loaded at startup.
	m.restoreResidents()
	if s := states(m); s["r"] != Ready {
		t.Fatalf("resident not preloaded: %v", s)
	}

	// Never unloaded for idleness.
	m.reapOnce(time.Now().Add(time.Hour))
	if s := states(m); s["r"] != Ready {
		t.Fatalf("resident reaped by TTL: %v", s)
	}

	// Gives way when another model needs the memory.
	_, rel, err := m.Acquire(ctx, "x")
	if err != nil {
		t.Fatal(err)
	}
	if s := states(m); s["r"] != Stopped || s["x"] != Ready {
		t.Fatalf("resident should be evicted for x: %v", s)
	}

	// Not restored while x is loaded (that would evict x).
	m.restoreResidents()
	if s := states(m); s["r"] != Stopped || s["x"] != Ready {
		t.Fatalf("restore must not evict x: %v", s)
	}
	rel()

	// x idles out, then the resident comes back.
	m.reapOnce(time.Now().Add(2 * time.Minute))
	m.restoreResidents()
	if s := states(m); s["r"] != Ready || s["x"] != Stopped {
		t.Fatalf("resident should be restored after x unloads: %v", s)
	}
	if want := []string{"r", "x", "r"}; len(r.started) != 3 || r.started[2] != "r" {
		t.Fatalf("starts = %v, want %v", r.started, want)
	}
}

func TestResidentHeldAfterManualUnload(t *testing.T) {
	m, _ := newTestManager(t, residentCfg)
	ctx := context.Background()
	m.restoreResidents()

	if err := m.Unload(ctx, "r"); err != nil {
		t.Fatal(err)
	}
	m.restoreResidents()
	if s := states(m); s["r"] != Stopped {
		t.Fatalf("manually unloaded resident came back: %v", s)
	}

	// Loading it again clears the hold, so later evictions restore it.
	if err := m.Load(ctx, "r"); err != nil {
		t.Fatal(err)
	}
	_, rel, err := m.Acquire(ctx, "x")
	if err != nil {
		t.Fatal(err)
	}
	rel()
	if err := m.Unload(ctx, "x"); err != nil {
		t.Fatal(err)
	}
	m.restoreResidents()
	if s := states(m); s["r"] != Ready {
		t.Fatalf("resident should be restored once the hold is cleared: %v", s)
	}
}

func TestResidentRestoreYieldsToQueuedClient(t *testing.T) {
	m, r := newTestManager(t, residentCfg)
	m.waiting.Add(1) // a client is queued for the lifecycle lock
	m.restoreResidents()
	m.waiting.Add(-1)
	if len(r.started) != 0 {
		t.Fatalf("restore ran ahead of a queued client: %v", r.started)
	}
}
