package stats

import "testing"

func TestParseUsageOpenAIStreamWithTimings(t *testing.T) {
	tail := []byte(`ing"}}]}

data: {"choices":[],"usage":{"completion_tokens":36,"prompt_tokens":57,"total_tokens":93,"prompt_tokens_details":{"cached_tokens":12}},"timings":{"cache_n":12,"prompt_n":45,"prompt_per_second":250.5,"predicted_n":36,"predicted_per_second":49.7,"draft_n":20,"draft_n_accepted":15}}

data: [DONE]

`)
	var s Sample
	ParseUsage(tail, &s)
	if s.PromptTokens != 57 || s.CompletionTokens != 36 || s.CachedTokens != 12 {
		t.Fatalf("tokens: %+v", s)
	}
	if s.GenTPS != 49.7 || s.GenTPSSource != "backend" || s.PromptTPS != 250.5 {
		t.Fatalf("timings: %+v", s)
	}
	if s.DraftTokens != 20 || s.DraftAccepted != 15 {
		t.Fatalf("draft: %+v", s)
	}
}

func TestParseUsageAnthropicStream(t *testing.T) {
	tail := []byte(`event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":120,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}

`)
	var s Sample
	ParseUsage(tail, &s)
	if s.PromptTokens != 120 || s.CompletionTokens != 42 {
		t.Fatalf("got %+v", s)
	}
	if s.GenTPS != 0 {
		t.Fatalf("no timings, gen tps should be left for the caller to measure: %+v", s)
	}
}

func TestParseUsageSkipsNullAndTruncated(t *testing.T) {
	// Starts mid-document; an early chunk has "usage":null.
	tail := []byte(`5,"x":1}} data: {"usage":null} data: {"usage":{"prompt_tokens":3,"completion_tokens":4}}`)
	var s Sample
	ParseUsage(tail, &s)
	if s.PromptTokens != 3 || s.CompletionTokens != 4 {
		t.Fatalf("got %+v", s)
	}
}

func TestRecorderSummary(t *testing.T) {
	r := NewRecorder()
	r.Observe("m", Sample{Status: 200, PromptTokens: 10, CompletionTokens: 20, GenTPS: 40, TTFTSeconds: 1, DraftTokens: 10, DraftAccepted: 8})
	r.Observe("m", Sample{Status: 200, PromptTokens: 5, CompletionTokens: 10, GenTPS: 60, TTFTSeconds: 3, DraftTokens: 10, DraftAccepted: 6})
	r.Observe("m", Sample{Status: 503})
	s := r.Summary("m")
	if s.Requests != 3 || s.Errors != 1 || s.PromptTokens != 15 || s.CompletionTokens != 30 {
		t.Fatalf("counts: %+v", s)
	}
	if s.AvgGenTPS != 50 || s.AvgTTFTSeconds != 2 || s.DraftAcceptRate != 0.7 {
		t.Fatalf("averages: %+v", s)
	}
	if s.Last == nil || s.Last.GenTPS != 60 || len(s.Recent) != 2 {
		t.Fatalf("last/recent: %+v", s)
	}
	if got := r.Summary("other"); got.Requests != 0 {
		t.Fatalf("unknown model: %+v", got)
	}
}
