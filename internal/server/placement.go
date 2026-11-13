package server

import (
	"net/http"
	"net/url"

	"github.com/hofftodd/ai-swap/internal/scheduler"
)

// candidate is one machine that can serve a requested model.
type candidate struct {
	node     string
	local    bool
	peer     *url.URL // nil when local
	model    string   // the model's name on that machine
	state    scheduler.State
	inflight int
	memoryGB float64
	freeGB   float64 // the machine's memory budget minus what its loaded models use
	reserved bool    // held by a reservation this request doesn't carry
}

func (c candidate) loaded() bool {
	return c.state == scheduler.Ready || c.state == scheduler.Starting
}

// fits reports whether loading the model needs no other model unloaded first.
func (c candidate) fits() bool {
	return c.memoryGB > 0 && c.memoryGB <= c.freeGB
}

// place picks where to serve a model that more than one machine has, so a
// request avoids making a machine swap models when another can serve it as is:
//
//  1. a machine where the model is loaded (or loading): fewest requests in
//     flight, then this machine;
//  2. a machine where it fits in free memory, so nothing is unloaded: this
//     machine first;
//  3. otherwise this machine, as before (unloading idle models there), or the
//     first peer if this machine doesn't have the model.
//
// Reserved machines come last: their requests would be refused, so one is
// picked only when every copy is reserved (and the caller gets the reason).
func place(cands []candidate) (candidate, bool) {
	if len(cands) == 0 {
		return candidate{}, false
	}
	better := func(a, b candidate) bool { // is a a better pick than b?
		if a.reserved != b.reserved {
			return !a.reserved
		}
		if a.loaded() != b.loaded() {
			return a.loaded()
		}
		if a.loaded() && a.inflight != b.inflight {
			return a.inflight < b.inflight
		}
		if !a.loaded() && a.fits() != b.fits() {
			return a.fits()
		}
		return a.local && !b.local
	}
	best := cands[0]
	for _, c := range cands[1:] {
		if better(c, best) {
			best = c
		}
	}
	return best, true
}

// candidates lists every machine that can serve name: this one (if it has the
// model and can run its engine here) and, when allowed, each online peer.
func (s *Server) candidates(r *http.Request, name string, allowPeers bool) []candidate {
	var out []candidate
	if m, ok := s.cfg.Resolve(name); ok {
		if inst, _ := s.mgr.Get(m.Name); inst != nil && inst.Engine.Supported() {
			c := candidate{
				node:     s.cfg.Node.Name,
				local:    true,
				model:    m.Name,
				memoryGB: m.MemoryGB,
				freeGB:   s.cfg.Node.MemoryBudgetGB - s.mgr.UsedGB(),
				reserved: s.reservedFor(r),
			}
			for _, st := range s.mgr.Snapshot() {
				if st.Name == m.Name {
					c.state, c.inflight = st.State, st.Inflight
				}
			}
			out = append(out, c)
		}
	}
	if allowPeers {
		for _, p := range s.cluster.Copies(name) {
			out = append(out, candidate{
				node:     p.Node,
				peer:     p.URL,
				model:    p.Model.Name,
				state:    p.Model.State,
				inflight: p.Model.Inflight,
				memoryGB: p.Model.MemoryGB,
				freeGB:   p.FreeGB,
				reserved: p.Lock != nil,
			})
		}
	}
	return out
}

// reservedFor reports whether this machine is reserved by someone other than
// the sender of r.
func (s *Server) reservedFor(r *http.Request) bool {
	if s.currentLock() == nil {
		return false
	}
	s.lock.mu.Lock()
	defer s.lock.mu.Unlock()
	return s.lock.token == "" || r.Header.Get(LockHeader) != s.lock.token
}

// holdsLock reports whether r carries this machine's reservation token.
func (s *Server) holdsLock(r *http.Request) bool {
	tok := r.Header.Get(LockHeader)
	if tok == "" || s.currentLock() == nil {
		return false
	}
	s.lock.mu.Lock()
	defer s.lock.mu.Unlock()
	return tok == s.lock.token
}
