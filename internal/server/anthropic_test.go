package server

import (
	"encoding/json"
	"strings"
	"testing"
)

type msg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

func fold(t *testing.T, in string) (system json.RawMessage, msgs []msg) {
	t.Helper()
	var req struct {
		System   json.RawMessage `json:"system"`
		Messages []msg           `json:"messages"`
	}
	if err := json.Unmarshal(foldSystemMessages([]byte(in)), &req); err != nil {
		t.Fatal(err)
	}
	for _, m := range req.Messages {
		if m.Role == "system" {
			t.Fatalf("system message left in messages: %s", m.Content)
		}
	}
	return req.System, req.Messages
}

func TestFoldClaudeCodeShape(t *testing.T) {
	// What Claude Code sends: a system message after the first user turn.
	_, msgs := fold(t, `{"model":"q","system":[{"type":"text","text":"sys"}],"messages":[
		{"role":"user","content":[{"type":"text","text":"hi"}]},
		{"role":"system","content":[{"type":"text","text":"# Environment"}]}]}`)
	if len(msgs) != 1 || msgs[0].Role != "user" {
		t.Fatalf("want one user message, got %+v", msgs)
	}
	var blocks []struct{ Text string }
	if err := json.Unmarshal(msgs[0].Content, &blocks); err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 2 || blocks[0].Text != "hi" || blocks[1].Text != "<system-reminder>\n# Environment\n</system-reminder>" {
		t.Fatalf("reminder should be appended after the user text: %s", msgs[0].Content)
	}
}

func TestFoldAfterToolUseGoesToNextUserTurn(t *testing.T) {
	// Inserting a user turn between tool_use and tool_result would break the
	// pairing, so the note joins the end of the following user turn.
	_, msgs := fold(t, `{"messages":[
		{"role":"user","content":"run it"},
		{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"n":12345678901234567890}}]},
		{"role":"system","content":"note"},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}]}`)
	if len(msgs) != 3 || msgs[1].Role != "assistant" || msgs[2].Role != "user" {
		t.Fatalf("roles: %+v", msgs)
	}
	c := string(msgs[2].Content)
	if !strings.HasPrefix(c, `[{"type":"tool_result"`) || !strings.Contains(c, "note") {
		t.Fatalf("tool_result must stay first, note appended: %s", c)
	}
	if !strings.Contains(string(msgs[1].Content), "12345678901234567890") {
		t.Fatalf("large integer in tool input was altered: %s", msgs[1].Content)
	}
}

func TestFoldLeadingSystemIntoSystemPrompt(t *testing.T) {
	sys, msgs := fold(t, `{"system":"base","messages":[
		{"role":"system","content":"extra"},{"role":"user","content":"hi"}]}`)
	var s string
	if json.Unmarshal(sys, &s) != nil || s != "base\n\nextra" {
		t.Fatalf("system = %s", sys)
	}
	if len(msgs) != 1 || string(msgs[0].Content) != `"hi"` {
		t.Fatalf("user turn should be untouched: %+v", msgs)
	}

	sys, _ = fold(t, `{"system":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}],"messages":[
		{"role":"system","content":"b"},{"role":"user","content":"hi"}]}`)
	if !strings.Contains(string(sys), `"cache_control":{"type":"ephemeral"}`) || !strings.Contains(string(sys), `"text":"b"`) {
		t.Fatalf("block system prompt: %s", sys)
	}
}

func TestFoldTrailingSystemBecomesUserTurn(t *testing.T) {
	_, msgs := fold(t, `{"messages":[
		{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"system","content":"late"}]}`)
	if len(msgs) != 3 || msgs[2].Role != "user" || !strings.Contains(string(msgs[2].Content), "late") {
		t.Fatalf("got %+v", msgs)
	}
}

func TestFoldNoSystemIsUnchanged(t *testing.T) {
	in := `{"model":"q","messages":[{"role":"user","content":"hi"}],"max_tokens":5}`
	if out := string(foldSystemMessages([]byte(in))); out != in {
		t.Fatalf("body changed: %s", out)
	}
	if out := string(foldSystemMessages([]byte("not json"))); out != "not json" {
		t.Fatalf("bad body changed: %s", out)
	}
}
