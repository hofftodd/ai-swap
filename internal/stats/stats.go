// Package stats measures the requests ai-swap proxies (time to first token,
// throughput, token counts) and gathers live details from backends, the
// engine process and the GPU, for the UI's model stats pane.
package stats

import (
	"bytes"
	"encoding/json"
	"sync"
	"time"
)

// window is how many recent requests the averages cover.
const window = 20

// Sample is one finished inference request.
type Sample struct {
	At               time.Time `json:"at"`
	Path             string    `json:"path"`
	Stream           bool      `json:"stream"`
	Status           int       `json:"status"`
	TTFTSeconds      float64   `json:"ttft_seconds"`  // request start to first response byte
	TotalSeconds     float64   `json:"total_seconds"` // request start to last byte
	PromptTokens     int       `json:"prompt_tokens,omitempty"`
	CachedTokens     int       `json:"cached_tokens,omitempty"`
	CompletionTokens int       `json:"completion_tokens,omitempty"`
	// PromptTPS / GenTPS are the backend's own figures when it reports
	// them (llama.cpp "timings"); otherwise GenTPS is estimated from
	// completion tokens over the streaming time.
	PromptTPS     float64 `json:"prompt_tps,omitempty"`
	GenTPS        float64 `json:"gen_tps,omitempty"`
	GenTPSSource  string  `json:"gen_tps_source,omitempty"` // "backend" or "measured"
	DraftTokens   int     `json:"draft_tokens,omitempty"`   // speculative decoding
	DraftAccepted int     `json:"draft_accepted,omitempty"`
}

// Summary is what the UI shows for one model.
type Summary struct {
	Requests         int       `json:"requests"`
	Errors           int       `json:"errors"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	AvgGenTPS        float64   `json:"avg_gen_tps,omitempty"`
	AvgPromptTPS     float64   `json:"avg_prompt_tps,omitempty"`
	AvgTTFTSeconds   float64   `json:"avg_ttft_seconds,omitempty"`
	DraftAcceptRate  float64   `json:"draft_accept_rate,omitempty"`
	Last             *Sample   `json:"last,omitempty"`
	Recent           []float64 `json:"recent_gen_tps,omitempty"` // oldest first, for a sparkline
}

type model struct {
	requests, errors    int
	promptTok, complTok int
	draftN, draftOK     int
	recent              []Sample // ring of the last `window` successful samples
}

// Recorder keeps per-model request statistics in memory.
type Recorder struct {
	mu     sync.Mutex
	models map[string]*model
}

// NewRecorder returns an empty recorder.
func NewRecorder() *Recorder { return &Recorder{models: map[string]*model{}} }

// Observe records one request for a model.
func (r *Recorder) Observe(name string, s Sample) {
	r.mu.Lock()
	defer r.mu.Unlock()
	m := r.models[name]
	if m == nil {
		m = &model{}
		r.models[name] = m
	}
	m.requests++
	if s.Status >= 400 {
		m.errors++
		return
	}
	m.promptTok += s.PromptTokens
	m.complTok += s.CompletionTokens
	m.draftN += s.DraftTokens
	m.draftOK += s.DraftAccepted
	m.recent = append(m.recent, s)
	if len(m.recent) > window {
		m.recent = m.recent[len(m.recent)-window:]
	}
}

// Summary returns the statistics for a model (zero value if none yet).
func (r *Recorder) Summary(name string) Summary {
	r.mu.Lock()
	defer r.mu.Unlock()
	m := r.models[name]
	if m == nil {
		return Summary{}
	}
	out := Summary{
		Requests: m.requests, Errors: m.errors,
		PromptTokens: m.promptTok, CompletionTokens: m.complTok,
	}
	if m.draftN > 0 {
		out.DraftAcceptRate = float64(m.draftOK) / float64(m.draftN)
	}
	var gen, prompt, ttft []float64
	for _, s := range m.recent {
		if s.GenTPS > 0 {
			gen = append(gen, s.GenTPS)
		}
		if s.PromptTPS > 0 {
			prompt = append(prompt, s.PromptTPS)
		}
		if s.TTFTSeconds > 0 {
			ttft = append(ttft, s.TTFTSeconds)
		}
	}
	out.AvgGenTPS, out.AvgPromptTPS, out.AvgTTFTSeconds = mean(gen), mean(prompt), mean(ttft)
	out.Recent = gen
	if n := len(m.recent); n > 0 {
		last := m.recent[n-1]
		out.Last = &last
	}
	return out
}

func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	var t float64
	for _, x := range v {
		t += x
	}
	return t / float64(len(v))
}

// ParseUsage fills token counts and backend timings into s from the tail of
// a response body. It understands OpenAI ("usage", streaming or not),
// Anthropic ("usage" with input/output tokens) and llama.cpp ("timings").
// The tail may start mid-document; only the last "usage" and "timings"
// objects are decoded.
func ParseUsage(tail []byte, s *Sample) {
	// Anthropic streams split usage: input tokens in message_start, output
	// tokens in message_delta. Merge every usage object, keeping the max.
	eachObject(tail, "usage", func(dec *json.Decoder) {
		var u struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			InputTokens      int `json:"input_tokens"`
			OutputTokens     int `json:"output_tokens"`
			CacheRead        int `json:"cache_read_input_tokens"`
			Details          struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		}
		if dec.Decode(&u) != nil {
			return
		}
		s.PromptTokens = max(s.PromptTokens, u.PromptTokens, u.InputTokens)
		s.CompletionTokens = max(s.CompletionTokens, u.CompletionTokens, u.OutputTokens)
		s.CachedTokens = max(s.CachedTokens, u.Details.CachedTokens, u.CacheRead)
	})
	var t struct {
		PromptPerSecond    float64 `json:"prompt_per_second"`
		PredictedPerSecond float64 `json:"predicted_per_second"`
		PredictedN         int     `json:"predicted_n"`
		CacheN             int     `json:"cache_n"`
		DraftN             int     `json:"draft_n"`
		DraftAccepted      int     `json:"draft_n_accepted"`
	}
	if lastObject(tail, "timings", &t) {
		s.PromptTPS = t.PromptPerSecond
		s.GenTPS, s.GenTPSSource = t.PredictedPerSecond, "backend"
		s.DraftTokens, s.DraftAccepted = t.DraftN, t.DraftAccepted
		if s.CompletionTokens == 0 {
			s.CompletionTokens = t.PredictedN
		}
		if s.CachedTokens == 0 {
			s.CachedTokens = t.CacheN
		}
	}
}

// eachObject calls fn with a decoder positioned at each JSON object that
// follows `"key":` in b.
func eachObject(b []byte, key string, fn func(*json.Decoder)) {
	needle := []byte(`"` + key + `":`)
	for {
		i := bytes.Index(b, needle)
		if i < 0 {
			return
		}
		b = b[i+len(needle):]
		rest := bytes.TrimLeft(b, " \t")
		if len(rest) > 0 && rest[0] == '{' {
			fn(json.NewDecoder(bytes.NewReader(rest)))
		}
	}
}

// lastObject decodes the JSON object following the last `"key":` in b.
func lastObject(b []byte, key string, v any) bool {
	needle := []byte(`"` + key + `":`)
	for end := len(b); end > 0; {
		i := bytes.LastIndex(b[:end], needle)
		if i < 0 {
			return false
		}
		rest := bytes.TrimLeft(b[i+len(needle):], " \t")
		if len(rest) > 0 && rest[0] == '{' && json.NewDecoder(bytes.NewReader(rest)).Decode(v) == nil {
			return true
		}
		end = i // e.g. "usage":null in an early chunk; keep looking further back
	}
	return false
}
