# Changelog

## Unreleased

- Claude Code works with strict chat templates (e.g. Qwen 3.8): mid-conversation
  `system` messages in `/v1/messages` requests are folded into the adjacent
  user turn as `<system-reminder>` blocks before reaching a local engine.

- `node.resident`: models a node keeps loaded by default. Loaded at
  startup, exempt from the idle TTL, evicted LRU-first under memory
  pressure, and reloaded automatically once they fit again.
- Stats pane in the UI and `GET /api/models/{name}/stats`: per-model request
  stats measured by the proxy (tok/s, TTFT, tokens, draft acceptance), live
  engine details (llama.cpp `/props`, `/slots`, `/metrics`, `/v1/models`),
  process RSS and AMD GPU memory. The llamacpp adapter now passes `--metrics`.
- UI rework after a usability review: phone and narrow-laptop layouts with
  dismissable drawers; a top bar showing the current model and state (the
  model dropdown and the desktop ☰ are gone); a filterable, per-machine model
  list with memory bars in group headers; a note before a send that will swap
  models, and "Loading… N s" while it does; Markdown tables, lists, headings,
  links and code blocks with copy; streaming that no longer forces the scroll
  position or reopens the thinking block; partial replies kept on errors and
  reloads; Copy / Retry / Edit on messages; a compact stats pane with
  collapsed details; logs in a dialog; an offline banner; keyboard focus kept
  across refreshes; a favicon.
- Fix: the chat UI failed over plain http from another machine
  (`crypto.randomUUID` is only available in secure contexts).

## v0.1.0 — 2026-11-04

First release, under the MIT license.

- OpenAI- and Anthropic-compatible proxy that routes requests by `model`,
  with SSE streaming.
- Engine adapters: `sushi`, `ds4`, `strata`, `llamacpp`, `generic`.
- Scheduler: memory-budget LRU eviction, fixed-port conflict handling,
  waiting for busy models, idle TTL, crash detection and restart.
- Multiple machines: peer polling, forwarding, `model@node` pinning, loop
  prevention, merged `/v1/models`. A peer is re-polled right after a
  request or load/unload is forwarded to it, so the UI updates at once.
- Management API: status, load, unload, logs.
- Embedded chat UI with streaming, a collapsible thinking section, tokens/sec,
  memory per machine, and a model control panel.
- `-check` to validate a config and print launch commands; `-version`.
- Example configs for an Apple Silicon Mac and a Linux box with a Radeon AI
  PRO R9700.
