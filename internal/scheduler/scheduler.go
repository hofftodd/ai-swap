// Package scheduler decides which models are loaded on this node. It starts
// models on demand, evicts idle ones when the memory budget (or a fixed
// port) requires it, and unloads models after their idle TTL.
package scheduler

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hofftodd/ai-swap/internal/config"
	"github.com/hofftodd/ai-swap/internal/engine"
	"github.com/hofftodd/ai-swap/internal/process"
)

// State is the lifecycle state of a model instance.
type State string

const (
	Stopped  State = "stopped"
	Starting State = "starting"
	Ready    State = "ready"
	Stopping State = "stopping"
	Failed   State = "failed"
)

const stopGrace = 15 * time.Second

// Handle is a started server process.
type Handle interface {
	WaitHealthy(ctx context.Context, timeout time.Duration) error
	Stop(grace time.Duration)
	Done() <-chan struct{}
}

// StartFunc starts a process; replaced in tests.
type StartFunc func(l *engine.Launch, logs *process.RingBuffer) (Handle, error)

func defaultStart(l *engine.Launch, logs *process.RingBuffer) (Handle, error) {
	return process.Start(l, logs)
}

// Instance is the runtime state of one configured model.
type Instance struct {
	Model  *config.Model
	Engine engine.Engine
	Logs   *process.RingBuffer

	state    State
	handle   Handle
	port     int
	inflight int
	lastUsed time.Time
	lastErr  string
	// held is set when a resident model was unloaded by hand; it is not
	// brought back automatically until something loads it again.
	held bool
	// loadedAt and loadDur describe the current (or last) successful start.
	loadedAt time.Time
	loadDur  time.Duration
	pid      int
	cmd      string // command line of the running process
}

// Status is a snapshot of an instance for the API/UI.
type Status struct {
	Name     string      `json:"name"`
	Engine   string      `json:"engine"`
	State    State       `json:"state"`
	Port     int         `json:"port,omitempty"`
	MemoryGB float64     `json:"memory_gb"`
	Ctx      int         `json:"ctx,omitempty"` // configured context length; 0 = engine default
	Inflight int         `json:"inflight"`
	LastUsed *time.Time  `json:"last_used,omitempty"`
	Error    string      `json:"error,omitempty"`
	Aliases  []string    `json:"aliases,omitempty"`
	Caps     engine.Caps `json:"caps"`
	Runnable bool        `json:"runnable"`
	Resident bool        `json:"resident,omitempty"`
	// LoadedAt is when the model became ready; LoadSeconds is how long that took.
	LoadedAt    *time.Time `json:"loaded_at,omitempty"`
	LoadSeconds float64    `json:"load_seconds,omitempty"`
	PID         int        `json:"pid,omitempty"`
	// Cmd is the running process's command line.
	Cmd string `json:"cmd,omitempty"`
}

// pider is implemented by handles that know their OS process ID.
type pider interface{ PID() int }

// Manager owns all model instances on this node.
type Manager struct {
	cfg   *config.Config
	start StartFunc
	log   *slog.Logger

	mu        sync.Mutex
	instances map[string]*Instance
	changed   chan struct{} // closed and replaced whenever state changes

	// lifecycle serialises every start/stop so the memory plan is never
	// computed against a half-applied previous plan.
	lifecycle chan struct{}
	// waiting counts callers queued for the lifecycle lock, so background
	// resident restores never jump ahead of a client's swap.
	waiting atomic.Int32
	paused  atomic.Bool // see SetPaused
}

// New builds a manager for the models in cfg.
func New(cfg *config.Config, log *slog.Logger) (*Manager, error) {
	m := &Manager{
		cfg:       cfg,
		start:     defaultStart,
		log:       log,
		instances: map[string]*Instance{},
		changed:   make(chan struct{}),
		lifecycle: make(chan struct{}, 1),
	}
	for _, name := range cfg.ModelNames() {
		mc := cfg.Models[name]
		e, err := engine.Get(mc.Engine)
		if err != nil {
			return nil, fmt.Errorf("model %s: %w", name, err)
		}
		m.instances[name] = &Instance{
			Model: mc, Engine: e, state: Stopped,
			Logs: process.NewRingBuffer(256 << 10),
		}
	}
	return m, nil
}

// SetStartFunc replaces the process starter (for tests).
func (m *Manager) SetStartFunc(f StartFunc) { m.start = f }

// Get returns the instance for a model name.
func (m *Manager) Get(name string) (*Instance, bool) {
	inst, ok := m.instances[name]
	return inst, ok
}

// Acquire ensures the model is loaded and marks one request in flight.
// The returned port is where the backend listens; release must be called
// when the request finishes.
func (m *Manager) Acquire(ctx context.Context, name string) (port int, release func(), err error) {
	inst, ok := m.instances[name]
	if !ok {
		return 0, nil, fmt.Errorf("unknown model %q", name)
	}
	if !inst.Engine.Supported() {
		return 0, nil, fmt.Errorf("model %q: engine %s does not run on this OS", name, inst.Engine.Name())
	}
	for {
		m.mu.Lock()
		if inst.state == Ready {
			inst.inflight++
			inst.lastUsed = time.Now()
			port := inst.port
			m.mu.Unlock()
			var once sync.Once
			return port, func() { once.Do(func() { m.release(inst) }) }, nil
		}
		m.mu.Unlock()
		if err := m.Load(ctx, name); err != nil {
			return 0, nil, err
		}
	}
}

func (m *Manager) release(inst *Instance) {
	m.mu.Lock()
	inst.inflight--
	inst.lastUsed = time.Now()
	m.notifyLocked()
	m.mu.Unlock()
}

// Load starts a model if it is not already ready, evicting others as needed.
func (m *Manager) Load(ctx context.Context, name string) error {
	inst, ok := m.instances[name]
	if !ok {
		return fmt.Errorf("unknown model %q", name)
	}
	if err := m.lock(ctx); err != nil {
		return err
	}
	defer m.unlock()

	for {
		m.mu.Lock()
		if inst.state == Ready {
			m.mu.Unlock()
			return nil
		}
		victims, wait := m.planLocked(inst)
		if wait {
			ch := m.changed
			m.mu.Unlock()
			m.log.Info("waiting for busy models to finish before swap", "model", name)
			select {
			case <-ch:
				continue
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		m.mu.Unlock()
		if len(victims) == 0 {
			break
		}
		for _, v := range victims {
			m.log.Info("evicting", "model", v.Model.Name, "for", name)
			m.stop(v)
		}
	}
	return m.startLocked(inst)
}

// Unload stops a model regardless of in-flight requests.
func (m *Manager) Unload(ctx context.Context, name string) error {
	inst, ok := m.instances[name]
	if !ok {
		return fmt.Errorf("unknown model %q", name)
	}
	if err := m.lock(ctx); err != nil {
		return err
	}
	defer m.unlock()
	m.mu.Lock()
	inst.held = m.cfg.IsResident(name)
	m.mu.Unlock()
	m.stop(inst)
	return nil
}

// StopAll stops every model (used on shutdown).
func (m *Manager) StopAll() {
	m.lifecycle <- struct{}{}
	defer m.unlock()
	var wg sync.WaitGroup
	for _, inst := range m.instances {
		wg.Add(1)
		go func() { defer wg.Done(); m.stop(inst) }()
	}
	wg.Wait()
}

// SetPaused stops (true) or resumes (false) idle unloads and resident
// reloads, e.g. while the machine is reserved for a benchmark. Resuming
// also clears the "unloaded by hand" mark on residents, so the ones the
// reservation holder unloaded come back.
func (m *Manager) SetPaused(p bool) {
	m.paused.Store(p)
	if !p {
		m.mu.Lock()
		for _, inst := range m.instances {
			inst.held = false
		}
		m.notifyLocked()
		m.mu.Unlock()
	}
}

// RunReaper loads resident models, then until ctx is done unloads idle
// models past their TTL and brings evicted residents back when they fit.
func (m *Manager) RunReaper(ctx context.Context, every time.Duration) {
	m.restoreResidents()
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if m.paused.Load() {
				continue
			}
			m.reapOnce(time.Now())
			if ctx.Err() == nil {
				m.restoreResidents()
			}
		}
	}
}

// restoreResidents starts any stopped resident model that fits in memory
// without evicting anything. It never waits: if a swap is in progress or a
// client is queued for one, it tries again on the next tick.
func (m *Manager) restoreResidents() {
	for _, name := range m.cfg.Node.Resident {
		if m.waiting.Load() > 0 {
			return
		}
		select {
		case m.lifecycle <- struct{}{}:
		default:
			return
		}
		inst := m.instances[name]
		m.mu.Lock()
		start := inst.state == Stopped && !inst.held && inst.Engine.Supported()
		if start {
			victims, wait := m.planLocked(inst)
			start = len(victims) == 0 && !wait
		}
		m.mu.Unlock()
		if start {
			m.log.Info("loading resident model", "model", name)
			_ = m.startLocked(inst) // failures are logged and leave it Failed
		}
		m.unlock()
	}
}

func (m *Manager) reapOnce(now time.Time) {
	select {
	case m.lifecycle <- struct{}{}:
	default:
		return // a swap is in progress; try next tick
	}
	defer m.unlock()
	var expired []*Instance
	m.mu.Lock()
	for _, inst := range m.instances {
		ttl := inst.Model.TTL.D()
		if m.cfg.IsResident(inst.Model.Name) {
			continue // residents give way to memory pressure, not idleness
		}
		if inst.state == Ready && inst.inflight == 0 && ttl > 0 && now.Sub(inst.lastUsed) > ttl {
			expired = append(expired, inst)
		}
	}
	m.mu.Unlock()
	for _, inst := range expired {
		m.log.Info("unloading idle model", "model", inst.Model.Name)
		m.stop(inst)
	}
}

// Snapshot returns the status of every instance, sorted by name.
func (m *Manager) Snapshot() []Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Status, 0, len(m.instances))
	for _, name := range m.cfg.ModelNames() {
		inst := m.instances[name]
		s := Status{
			Name: name, Engine: inst.Engine.Name(), State: inst.state,
			MemoryGB: inst.Model.MemoryGB, Ctx: inst.Model.Ctx, Inflight: inst.inflight,
			Error: inst.lastErr, Aliases: inst.Model.Aliases,
			Caps: inst.Engine.Caps(), Runnable: inst.Engine.Supported(),
			Resident: m.cfg.IsResident(name),
		}
		if inst.state == Ready || inst.state == Starting {
			s.Port = inst.port
			s.PID = inst.pid
			s.Cmd = inst.cmd
		}
		if inst.state == Ready && !inst.loadedAt.IsZero() {
			t := inst.loadedAt
			s.LoadedAt = &t
		}
		if inst.loadDur > 0 {
			s.LoadSeconds = inst.loadDur.Seconds()
		}
		if !inst.lastUsed.IsZero() {
			t := inst.lastUsed
			s.LastUsed = &t
		}
		out = append(out, s)
	}
	return out
}

// UsedGB returns the memory reserved by starting/ready models.
func (m *Manager) UsedGB() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	var used float64
	for _, inst := range m.instances {
		if active(inst.state) {
			used += inst.Model.MemoryGB
		}
	}
	return used
}

func (m *Manager) lock(ctx context.Context) error {
	m.waiting.Add(1)
	defer m.waiting.Add(-1)
	select {
	case m.lifecycle <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *Manager) unlock() { <-m.lifecycle }

func (m *Manager) notifyLocked() {
	close(m.changed)
	m.changed = make(chan struct{})
}

func active(s State) bool { return s == Ready || s == Starting || s == Stopping }

// candidate is the scheduler's view of another instance.
type candidate struct {
	inst     *Instance
	memGB    float64
	port     int
	inflight int
	lastUsed time.Time
}

// planLocked works out which instances must stop before target can start.
// wait is true when the plan is blocked on models that are serving requests.
func (m *Manager) planLocked(target *Instance) (victims []*Instance, wait bool) {
	var others []candidate
	for _, inst := range m.instances {
		if inst == target || !active(inst.state) {
			continue
		}
		others = append(others, candidate{
			inst: inst, memGB: inst.Model.MemoryGB, port: inst.port,
			inflight: inst.inflight, lastUsed: inst.lastUsed,
		})
	}
	return plan(others, target.Model.MemoryGB, m.cfg.Node.MemoryBudgetGB, m.fixedPort(target))
}

// plan is the pure eviction policy: stop anything holding the target's fixed
// port, then least-recently-used idle models until the target fits.
func plan(others []candidate, needGB, budgetGB float64, fixedPort int) (victims []*Instance, wait bool) {
	evict := map[*Instance]bool{}
	for _, o := range others {
		if fixedPort != 0 && o.port == fixedPort {
			if o.inflight > 0 {
				wait = true
			} else {
				evict[o.inst] = true
			}
		}
	}
	if budgetGB > 0 {
		var used float64
		var idle []candidate
		for _, o := range others {
			if evict[o.inst] {
				continue
			}
			used += o.memGB
			if o.inflight == 0 {
				idle = append(idle, o)
			}
		}
		sort.Slice(idle, func(i, j int) bool { return idle[i].lastUsed.Before(idle[j].lastUsed) })
		for used+needGB > budgetGB && len(idle) > 0 {
			evict[idle[0].inst] = true
			used -= idle[0].memGB
			idle = idle[1:]
		}
		if used+needGB > budgetGB {
			wait = true
		}
	}
	if wait {
		return nil, true
	}
	for _, o := range others {
		if evict[o.inst] {
			victims = append(victims, o.inst)
		}
	}
	return victims, false
}

// fixedPort returns the port an instance will use if it cannot be moved.
func (m *Manager) fixedPort(inst *Instance) int {
	l, err := inst.Engine.Build(inst.Model, m.cfg.Engines[inst.Model.Engine], func() (int, error) {
		return 0, errNoFixedPort
	})
	if err != nil || l == nil {
		return 0
	}
	return l.Port
}

var errNoFixedPort = errors.New("dynamic port")

// startLocked starts inst; the caller holds the lifecycle lock.
func (m *Manager) startLocked(inst *Instance) error {
	name := inst.Model.Name
	l, err := inst.Engine.Build(inst.Model, m.cfg.Engines[inst.Model.Engine], m.allocPort)
	if err != nil {
		m.fail(inst, err)
		return err
	}
	m.mu.Lock()
	inst.state, inst.port, inst.lastErr, inst.cmd = Starting, l.Port, "", strings.Join(l.Argv, " ")
	m.notifyLocked()
	m.mu.Unlock()

	m.log.Info("starting", "model", name, "engine", inst.Engine.Name(), "port", l.Port)
	t0 := time.Now()
	h, err := m.start(l, inst.Logs)
	if err != nil {
		err = fmt.Errorf("start %s: %w", name, err)
		m.fail(inst, err)
		return err
	}
	if p, ok := h.(pider); ok {
		m.mu.Lock()
		inst.pid = p.PID()
		m.mu.Unlock()
	}
	// Loading is not tied to the request context: a client giving up after
	// a minute should not throw away a nearly-loaded 100 GB model.
	if err := h.WaitHealthy(context.Background(), inst.Model.HealthTimeout.D()); err != nil {
		h.Stop(stopGrace)
		err = fmt.Errorf("%s: %w", name, err)
		m.fail(inst, err)
		return err
	}

	m.mu.Lock()
	inst.handle, inst.state, inst.lastUsed, inst.held = h, Ready, time.Now(), false
	inst.loadedAt, inst.loadDur = time.Now(), time.Since(t0)
	m.notifyLocked()
	m.mu.Unlock()
	m.log.Info("ready", "model", name, "port", l.Port)

	go func() {
		<-h.Done()
		m.mu.Lock()
		defer m.mu.Unlock()
		if inst.handle == h && inst.state == Ready {
			inst.state, inst.handle, inst.lastErr = Failed, nil, "process exited unexpectedly"
			m.log.Warn("model process exited", "model", name)
			m.notifyLocked()
		}
	}()
	return nil
}

func (m *Manager) fail(inst *Instance, err error) {
	m.mu.Lock()
	inst.state, inst.lastErr, inst.handle = Failed, err.Error(), nil
	m.notifyLocked()
	m.mu.Unlock()
	m.log.Error("model failed", "model", inst.Model.Name, "err", err)
}

// stop stops inst; the caller holds the lifecycle lock.
func (m *Manager) stop(inst *Instance) {
	m.mu.Lock()
	h := inst.handle
	if h == nil {
		if inst.state != Failed {
			inst.state = Stopped
		}
		m.mu.Unlock()
		return
	}
	inst.state = Stopping
	m.notifyLocked()
	m.mu.Unlock()

	h.Stop(stopGrace)

	m.mu.Lock()
	inst.state, inst.handle, inst.pid = Stopped, nil, 0
	m.notifyLocked()
	m.mu.Unlock()
}

// allocPort returns a free port from the node's range not used by any instance.
func (m *Manager) allocPort() (int, error) {
	m.mu.Lock()
	used := map[int]bool{}
	for _, inst := range m.instances {
		if active(inst.state) {
			used[inst.port] = true
		}
	}
	m.mu.Unlock()
	lo, hi := m.cfg.Node.PortRange[0], m.cfg.Node.PortRange[1]
	for p := lo; p <= hi; p++ {
		if used[p] {
			continue
		}
		ln, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(p))
		if err != nil {
			continue
		}
		ln.Close()
		return p, nil
	}
	return 0, fmt.Errorf("no free port in %d-%d", lo, hi)
}
