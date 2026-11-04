# Changelog

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
