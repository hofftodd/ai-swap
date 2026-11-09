// Package cluster tracks peer ai-swap nodes and which models they serve, so
// a request for a model that lives on another machine can be forwarded.
package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/hofftodd/ai-swap/internal/config"
	"github.com/hofftodd/ai-swap/internal/hostinfo"
	"github.com/hofftodd/ai-swap/internal/scheduler"
)

// HopHeader marks a request already forwarded by a peer. A node that sees
// it serves only its own models, which prevents forwarding loops.
const HopHeader = "X-AI-Swap-Hop"

// NodeInfo is what each node publishes at /api/node.
type NodeInfo struct {
	Name           string             `json:"name"`
	Version        string             `json:"version"`
	OS             string             `json:"os"`
	Arch           string             `json:"arch"`
	MemoryBudgetGB float64            `json:"memory_budget_gb"`
	MemoryUsedGB   float64            `json:"memory_used_gb"`
	Role           string             `json:"role,omitempty"`
	Models         []scheduler.Status `json:"models"`
	// Hardware describes the machine: OS, CPU, memory, GPUs, power, temperatures.
	Hardware *hostinfo.Info `json:"hardware,omitempty"`
	// Lock is set while a client (e.g. a benchmark run) has reserved this machine.
	Lock *LockInfo `json:"lock,omitempty"`
}

// LockInfo describes a reservation of a machine (the token is never shown).
type LockInfo struct {
	Owner   string    `json:"owner"`
	Reason  string    `json:"reason,omitempty"`
	Models  []string  `json:"models,omitempty"`
	Since   time.Time `json:"since"`
	Expires time.Time `json:"expires"`
}

// PeerState is the last known state of a peer.
type PeerState struct {
	Name     string    `json:"name"`
	URL      string    `json:"url"`
	Online   bool      `json:"online"`
	Error    string    `json:"error,omitempty"`
	LastSeen time.Time `json:"last_seen,omitempty"`
	Node     *NodeInfo `json:"node,omitempty"`

	parsed *url.URL
}

// Cluster polls peers and answers "who serves model X?".
type Cluster struct {
	log    *slog.Logger
	client *http.Client

	mu    sync.RWMutex
	peers []*PeerState
}

// New creates a cluster view from the configured peers.
func New(peers []config.Peer, log *slog.Logger) (*Cluster, error) {
	c := &Cluster{log: log, client: &http.Client{Timeout: 5 * time.Second, Transport: NewTransport()}}
	for _, p := range peers {
		u, err := url.Parse(strings.TrimRight(p.URL, "/"))
		if err != nil {
			return nil, fmt.Errorf("peer %s: %w", p.Name, err)
		}
		c.peers = append(c.peers, &PeerState{Name: p.Name, URL: u.String(), parsed: u})
	}
	return c, nil
}

// Run polls every peer until ctx is done.
func (c *Cluster) Run(ctx context.Context, every time.Duration) {
	c.pollAll(ctx)
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			c.pollAll(ctx)
		}
	}
}

func (c *Cluster) pollAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, p := range c.peers {
		wg.Add(1)
		go func() { defer wg.Done(); c.poll(ctx, p) }()
	}
	wg.Wait()
}

func (c *Cluster) poll(ctx context.Context, p *PeerState) {
	info, err := c.fetch(ctx, p)
	c.mu.Lock()
	defer c.mu.Unlock()
	if err != nil {
		if p.Online {
			c.log.Warn("peer offline", "peer", p.Name, "err", err)
		}
		p.Online, p.Error = false, err.Error()
		return
	}
	if !p.Online {
		c.log.Info("peer online", "peer", p.Name, "node", info.Name, "models", len(info.Models))
	}
	p.Online, p.Error, p.LastSeen, p.Node = true, "", time.Now(), info
}

func (c *Cluster) fetch(ctx context.Context, p *PeerState) (*NodeInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.URL+"/api/node", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set(HopHeader, "1")
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %s", resp.Status)
	}
	var info NodeInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, err
	}
	return &info, nil
}

// Refresh re-polls the peer at base in the background, so state changes
// caused by a forwarded request (a model loading or unloading) show up
// without waiting for the next poll interval.
func (c *Cluster) Refresh(base *url.URL) {
	for _, p := range c.peers {
		if p.parsed.String() == base.String() {
			go func() {
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				c.poll(ctx, p)
			}()
			return
		}
	}
}

// NewTransport returns the HTTP transport used for peers and engines.
//
// It tries IPv4 first and falls back to any address family. On a LAN with
// mDNS names, IPv6 tends to be the broken half: a host that publishes IPv6
// addresses the network won't deliver hangs each new connection, and a host
// with IPv6 turned off makes macOS wait 5 s for an AAAA answer that never
// comes (longer than a peer poll allows). Asking for IPv4 alone avoids both.
//
// Connecting is bounded to 3 s, so a dead peer fails fast. There is no
// response-header timeout: prefill on a huge prompt can take minutes.
func NewTransport() *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	d := &net.Dialer{Timeout: 3 * time.Second, KeepAlive: 30 * time.Second}
	t.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		if network == "tcp" {
			if c, err := d.DialContext(ctx, "tcp4", addr); err == nil {
				return c, nil
			}
		}
		return d.DialContext(ctx, network, addr)
	}
	t.ResponseHeaderTimeout = 0
	t.MaxIdleConnsPerHost = 32 // several agents streaming through one peer
	return t
}

// MarkDown records that forwarding to the peer at base failed to connect, so
// later requests fail fast instead of each waiting for a timeout. The next
// successful poll brings the peer back.
func (c *Cluster) MarkDown(base *url.URL, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, p := range c.peers {
		if p.parsed.String() == base.String() && p.Online {
			c.log.Warn("peer offline", "peer", p.Name, "err", err)
			p.Online, p.Error = false, err.Error()
		}
	}
}

// FindOffline reports which offline peer last listed the model, so a client
// can be told the model is temporarily unreachable rather than unknown.
func (c *Cluster) FindOffline(name string) (peer string, lastSeen time.Time, ok bool) {
	want, node, pinned := strings.Cut(name, "@")
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, p := range c.peers {
		if p.Online || p.Node == nil {
			continue
		}
		if pinned && p.Name != node && p.Node.Name != node {
			continue
		}
		for _, m := range p.Node.Models {
			if m.Name == want || contains(m.Aliases, want) {
				return p.Name, p.LastSeen, true
			}
		}
	}
	return "", time.Time{}, false
}

// Peers returns a copy of every peer's state.
func (c *Cluster) Peers() []PeerState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]PeerState, len(c.peers))
	for i, p := range c.peers {
		out[i] = *p
	}
	return out
}

// Find returns the peer URL and the peer's own model name for a request.
// name may be a model, an alias, or "model@node" to pin a node.
// Online peers that can run the model are preferred, then ones with it ready.
func (c *Cluster) Find(name string) (target *url.URL, model string, ok bool) {
	want, node, pinned := strings.Cut(name, "@")
	c.mu.RLock()
	defer c.mu.RUnlock()
	var best *PeerState
	var bestModel string
	var bestReady bool
	for _, p := range c.peers {
		if !p.Online || p.Node == nil {
			continue
		}
		if pinned && p.Name != node && p.Node.Name != node {
			continue
		}
		for _, m := range p.Node.Models {
			if !m.Runnable || !(m.Name == want || contains(m.Aliases, want)) {
				continue
			}
			ready := m.State == scheduler.Ready
			if best == nil || (ready && !bestReady) {
				best, bestModel, bestReady = p, m.Name, ready
			}
		}
	}
	if best == nil {
		return nil, "", false
	}
	return best.parsed, bestModel, true
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
