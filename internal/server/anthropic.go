package server

import (
	"encoding/json"
	"strings"
)

// foldSystemMessages rewrites an Anthropic Messages request so it contains no
// "system"-role entries inside "messages".
//
// Claude Code (and the Anthropic API) allow system messages mid-conversation,
// but most local chat templates only accept a system prompt at the very start
// (Qwen's raises "System message must be at the beginning"). Each such message
// is folded, in place, into a neighbouring user turn as a <system-reminder>
// block: appended to the previous user message, else to the next one, else
// kept as its own user message. One at the very start, before any other turn,
// goes into the top-level "system" prompt instead. Folding in place keeps the
// earlier part of the prompt unchanged, so the engine's prompt cache still
// applies. The body is returned unchanged if there is nothing to fold or it
// cannot be parsed.
func foldSystemMessages(body []byte) []byte {
	var req map[string]json.RawMessage
	if json.Unmarshal(body, &req) != nil {
		return body
	}
	var msgs []map[string]json.RawMessage
	if json.Unmarshal(req["messages"], &msgs) != nil {
		return body
	}
	var role = func(m map[string]json.RawMessage) string {
		var r string
		_ = json.Unmarshal(m["role"], &r)
		return r
	}
	found := false
	for _, m := range msgs {
		if role(m) == "system" {
			found = true
			break
		}
	}
	if !found {
		return body
	}

	var out []map[string]json.RawMessage
	var leading []string   // system text before any other turn
	pending := [][]block{} // folded blocks waiting for the next user turn
	for _, m := range msgs {
		if role(m) != "system" {
			if role(m) == "user" && len(pending) > 0 {
				blocks := contentBlocks(m["content"])
				for _, p := range pending {
					blocks = append(blocks, p...)
				}
				m["content"], _ = json.Marshal(blocks)
				pending = pending[:0]
			}
			out = append(out, m)
			continue
		}
		text := textOf(m["content"])
		if strings.TrimSpace(text) == "" {
			continue
		}
		if len(out) == 0 {
			leading = append(leading, text)
			continue
		}
		reminder := []block{textBlock("<system-reminder>\n" + text + "\n</system-reminder>")}
		if prev := out[len(out)-1]; role(prev) == "user" {
			prev["content"], _ = json.Marshal(append(contentBlocks(prev["content"]), reminder...))
		} else {
			pending = append(pending, reminder)
		}
	}
	for _, p := range pending { // no later user turn: keep as its own message
		c, _ := json.Marshal(p)
		r, _ := json.Marshal("user")
		out = append(out, map[string]json.RawMessage{"role": r, "content": c})
	}
	if len(leading) > 0 {
		req["system"] = appendSystem(req["system"], strings.Join(leading, "\n\n"))
	}
	var err error
	if req["messages"], err = json.Marshal(out); err != nil {
		return body
	}
	b, err := json.Marshal(req)
	if err != nil {
		return body
	}
	return b
}

// block is one content block, kept as raw JSON so existing blocks (tool
// calls, images, cache_control...) pass through byte-for-byte.
type block = json.RawMessage

func textBlock(text string) block {
	b, _ := json.Marshal(struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}{"text", text})
	return b
}

// contentBlocks normalises Anthropic message content (a string or an array of
// blocks) to an array of blocks.
func contentBlocks(raw json.RawMessage) []block {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "" {
			return nil
		}
		return []block{textBlock(s)}
	}
	var blocks []block
	_ = json.Unmarshal(raw, &blocks)
	return blocks
}

// textOf joins the text of a string or block-array content.
func textOf(raw json.RawMessage) string {
	var parts []string
	for _, b := range contentBlocks(raw) {
		var t struct {
			Text string `json:"text"`
		}
		if json.Unmarshal(b, &t) == nil && t.Text != "" {
			parts = append(parts, t.Text)
		}
	}
	return strings.Join(parts, "\n\n")
}

// appendSystem adds text to the top-level system prompt (string or blocks).
func appendSystem(raw json.RawMessage, text string) json.RawMessage {
	var s string
	if len(raw) == 0 || string(raw) == "null" {
		out, _ := json.Marshal(text)
		return out
	}
	if json.Unmarshal(raw, &s) == nil {
		out, _ := json.Marshal(strings.TrimRight(s, "\n") + "\n\n" + text)
		return out
	}
	var blocks []block
	if json.Unmarshal(raw, &blocks) != nil {
		return raw
	}
	out, _ := json.Marshal(append(blocks, textBlock(text)))
	return out
}
