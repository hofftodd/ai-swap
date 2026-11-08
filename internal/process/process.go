// Package process runs one inference-server process: start, wait for it to
// become healthy, capture its logs, and stop it (including child processes).
package process

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/hofftodd/ai-swap/internal/engine"
)

// Process is a running (or exited) server process.
type Process struct {
	launch *engine.Launch
	cmd    *exec.Cmd
	logs   *RingBuffer
	done   chan struct{}
	err    error // set before done is closed
	once   sync.Once
}

// Start launches the server. It does not wait for readiness.
func Start(l *engine.Launch, logs *RingBuffer) (*Process, error) {
	if len(l.Argv) == 0 {
		return nil, errors.New("empty command")
	}
	cmd := exec.Command(l.Argv[0], l.Argv[1:]...)
	cmd.Dir = l.Dir
	cmd.Env = append(os.Environ(), l.Env...)
	cmd.Stdout = logs
	cmd.Stderr = logs
	// Own process group, so Stop also reaches wrapper scripts' children.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	fmt.Fprintf(logs, "[ai-swap] exec %q (port %d)\n", l.Argv, l.Port)
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	p := &Process{launch: l, cmd: cmd, logs: logs, done: make(chan struct{})}
	go func() {
		p.err = cmd.Wait()
		fmt.Fprintf(logs, "[ai-swap] process exited: %v\n", p.err)
		close(p.done)
	}()
	return p, nil
}

// Done is closed when the process exits.
func (p *Process) Done() <-chan struct{} { return p.done }

// Err is the exit error, valid after Done is closed.
func (p *Process) Err() error { return p.err }

// PID returns the process ID.
func (p *Process) PID() int { return p.cmd.Process.Pid }

// WaitHealthy polls the health endpoint until it answers 200, the process
// exits, or the timeout/context expires.
func (p *Process) WaitHealthy(ctx context.Context, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	url := fmt.Sprintf("http://127.0.0.1:%d%s", p.launch.Port, p.launch.HealthPath)
	client := &http.Client{Timeout: 5 * time.Second}
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if resp, err := client.Do(req); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return p.warmup(ctx)
			}
		}
		select {
		case <-p.done:
			return fmt.Errorf("exited before becoming healthy: %v", p.err)
		case <-ctx.Done():
			return fmt.Errorf("not healthy at %s: %w", url, ctx.Err())
		case <-tick.C:
		}
	}
}

// warmup POSTs the launch's WarmupPath, if any, and waits for a 2xx. Loading
// weights can take minutes, so it uses the remaining health timeout.
func (p *Process) warmup(ctx context.Context) error {
	if p.launch.WarmupPath == "" {
		return nil
	}
	url := fmt.Sprintf("http://127.0.0.1:%d%s", p.launch.Port, p.launch.WarmupPath)
	fmt.Fprintf(p.logs, "[ai-swap] warm-up POST %s\n", p.launch.WarmupPath)
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("warm-up %s: %w", p.launch.WarmupPath, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("warm-up %s: %s: %s", p.launch.WarmupPath, resp.Status, bytes.TrimSpace(msg))
	}
	return nil
}

// Stop sends SIGTERM to the process group, then SIGKILL after grace.
func (p *Process) Stop(grace time.Duration) {
	p.once.Do(func() {
		pgid := p.cmd.Process.Pid
		_ = syscall.Kill(-pgid, syscall.SIGTERM)
		select {
		case <-p.done:
		case <-time.After(grace):
			fmt.Fprintf(p.logs, "[ai-swap] no exit after %s, killing\n", grace)
			_ = syscall.Kill(-pgid, syscall.SIGKILL)
			<-p.done
		}
	})
}

// RingBuffer keeps the last N bytes written to it.
type RingBuffer struct {
	mu   sync.Mutex
	buf  []byte
	size int
}

// NewRingBuffer returns a buffer holding at most size bytes.
func NewRingBuffer(size int) *RingBuffer { return &RingBuffer{size: size} }

func (r *RingBuffer) Write(b []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.buf = append(r.buf, b...)
	if over := len(r.buf) - r.size; over > 0 {
		r.buf = append(r.buf[:0:0], r.buf[over:]...)
	}
	return len(b), nil
}

// Bytes returns a copy of the buffered output.
func (r *RingBuffer) Bytes() []byte {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]byte(nil), r.buf...)
}
