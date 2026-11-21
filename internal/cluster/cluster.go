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
	Name           string  `json:"name"`
	Version        string  `json:"version"`
	OS             string  `json:"os"`
	Arch           string  `json:"arch"`
	MemoryBudgetGB float64 `json:"memory_budget_gb"`
	MemoryUsedGB   float64 `json:"memory_used_gb"`
	Role           string  `json:"role,omitempty"`
	// DefaultModel serves requests that name no model (or "default").
	DefaultModel string             `json:"default_model,omitempty"`
	Models       []scheduler.Status `json:"models"`
	// Hardware describes the machine: OS, CPU, memory, GPUs, power, temperatures.
	Hardware *hostinfo.Info `json:"hardware,omitempty"`
	// Lock is set while a client (e.g. a benchmark run) has reserved this machine.
	Lock *LockInfo `json:"lock,omitempty"`
	// Leases are clients keeping a model loaded across many requests.
	Leases []LeaseInfo `json:"leases,omitempty"`
	// Peers are the other machines this node knows, so a node that lists only
	// one of them learns the rest.
	Peers []PeerRef `json:"peers,omitempty"`
}

// PeerRef names another node and where to reach it.
type PeerRef struct {
	Name string `json:"name"`
	URL  string `json:"url"`
}

// LockInfo describes a reservation of a machine (the token is never shown).
type LockInfo struct {
	Owner   string    `json:"owner"`
	Reason  string    `json:"reason,omitempty"`
	Models  []string  `json:"models,omitempty"`
	Since   time.Time `json:"since"`
	Expires time.Time `json:"expires"`
}

// LeaseInfo describes a lease on a model (the lease id is never shown).
type LeaseInfo struct {
	Model   string    `json:"model"`
	Owner   string    `json:"owner"`
	Reason  string    `json:"reason,omitempty"`
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
	// Learned is set for a peer found through another peer rather than
	// configured; it is dropped after learnedFails failed polls in a row.
	Learned bool `json:"learned,omitempty"`

	parsed *url.URL
	fails  int
}

// learnedFails is how many failed polls in a row drop a learned peer.
const learnedFails = 3

// Cluster polls peers and answers "who serves model X?".
type Cluster struct {
	log    *slog.Logger
	client *http.Client
	apiKey string // sent to peers; see SetAPIKey
	self   string // this node's name, never added as a peer

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

// SetSelf names this node, so a peer's list that includes it is ignored.
func (c *Cluster) SetSelf(name string) { c.self = name }

// SetAPIKey sets the key sent with every poll (the cluster shares one key).
func (c *Cluster) SetAPIKey(key string) { c.apiKey = key }

// APIKey is the key sent to peers ("" when the cluster has none).
func (c *Cluster) APIKey() string { return c.apiKey }

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

// snapshot returns the current peer list (it grows as peers are learned).
func (c *Cluster) snapshot() []*PeerState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return append([]*PeerState(nil), c.peers...)
}

func (c *Cluster) pollAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, p := range c.snapshot() {
		wg.Add(1)
		go func() { defer wg.Done(); c.poll(ctx, p) }()
	}
	wg.Wait()
}

func (c *Cluster) poll(ctx context.Context, p *PeerState) {
	info, err := c.fetch(ctx, p)
	c.mu.Lock()
	if err != nil {
		if p.Online {
			c.log.Warn("peer offline", "peer", p.Name, "err", err)
		}
		p.Online, p.Error = false, err.Error()
		p.fails++
		if p.Learned && p.fails >= learnedFails {
			c.dropLocked(p)
		}
		c.mu.Unlock()
		return
	}
	if !p.Online {
		c.log.Info("peer online", "peer", p.Name, "node", info.Name, "models", len(info.Models))
	}
	p.Online, p.Error, p.LastSeen, p.Node, p.fails = true, "", time.Now(), info, 0
	added := c.learnLocked(p, info.Peers)
	c.mu.Unlock()
	for _, n := range added {
		go c.poll(ctx, n)
	}
}

// learnLocked adds the peers that peer from knows and this node doesn't, and
// returns them so they can be polled at once. A loopback address is only
// meaningful if from is itself reached over loopback (the same machine).
// The caller holds c.mu.
func (c *Cluster) learnLocked(from *PeerState, refs []PeerRef) []*PeerState {
	var added []*PeerState
	for _, r := range refs {
		if r.Name == "" || r.Name == c.self || c.knownLocked(r) {
			continue
		}
		u, err := url.Parse(strings.TrimRight(r.URL, "/"))
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" ||
			(loopback(u) && !loopback(from.parsed)) || linkLocal(u) {
			continue
		}
		p := &PeerState{Name: r.Name, URL: u.String(), parsed: u, Learned: true}
		c.peers = append(c.peers, p)
		added = append(added, p)
		c.log.Info("peer learned", "peer", r.Name, "url", p.URL)
	}
	return added
}

func (c *Cluster) knownLocked(r PeerRef) bool {
	want := strings.TrimRight(r.URL, "/")
	for _, p := range c.peers {
		if p.Name == r.Name || p.URL == want || (p.Node != nil && p.Node.Name == r.Name) {
			return true
		}
	}
	return false
}

func (c *Cluster) dropLocked(gone *PeerState) {
	for i, p := range c.peers {
		if p == gone {
			c.peers = append(c.peers[:i], c.peers[i+1:]...)
			c.log.Warn("learned peer dropped", "peer", gone.Name, "fails", gone.fails)
			return
		}
	}
}

// loopback reports whether u points at the machine it's resolved on.
func loopback(u *url.URL) bool {
	h := u.Hostname()
	if h == "localhost" {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

// linkLocal reports a link-local address (such as a cloud metadata service),
// which is never a peer: the cluster key is sent to every peer it polls.
func linkLocal(u *url.URL) bool {
	ip := net.ParseIP(u.Hostname())
	return ip != nil && (ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified())
}

// Refs lists the peers to publish to other nodes: online ones, by node name.
func (c *Cluster) Refs() []PeerRef {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var out []PeerRef
	for _, p := range c.peers {
		if !p.Online || p.Node == nil {
			continue
		}
		out = append(out, PeerRef{Name: p.Node.Name, URL: p.URL})
	}
	return out
}

func (c *Cluster) fetch(ctx context.Context, p *PeerState) (*NodeInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.URL+"/api/node", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set(HopHeader, "1")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
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
	for _, p := range c.snapshot() {
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

// Copy is one online peer's copy of a model, with what placement needs to know.
type Copy struct {
	Node   string
	URL    *url.URL
	Model  scheduler.Status
	FreeGB float64   // the peer's memory budget minus what its loaded models use
	Lock   *LockInfo // the peer's reservation, if any
}

// Copies lists every online peer that can run the model (a name or an alias),
// as last polled.
func (c *Cluster) Copies(name string) []Copy {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var out []Copy
	for _, p := range c.peers {
		if !p.Online || p.Node == nil {
			continue
		}
		for _, m := range p.Node.Models {
			if m.Runnable && (m.Name == name || contains(m.Aliases, name)) {
				out = append(out, Copy{
					Node:   p.Name,
					URL:    p.parsed,
					Model:  m,
					FreeGB: p.Node.MemoryBudgetGB - p.Node.MemoryUsedGB,
					Lock:   p.Node.Lock,
				})
				break
			}
		}
	}
	return out
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
