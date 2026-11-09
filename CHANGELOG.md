# Changelog

## Unreleased

- Hardware: each machine reports what it runs on in `/api/node` (and so to its
  peers): OS and version, kernel, the computer's model, CPU (model, cores and
  threads or P/E cores, clock, utilisation, temperature, package power where
  readable), memory and swap, load, uptime, each GPU (VRAM, utilisation, power
  draw and limit, edge and junction temperature, fan, clock: amdgpu on Linux,
  the chip and GPU cores on Apple Silicon) and AC/battery. Read without root;
  what needs root (CPU power on recent kernels, power on macOS) says so. The
  Models view shows it under each machine (collapsible, remembered).
- `node.role`: a short note shown with the machine (e.g. "ai-factory host");
  machines with a role are listed first. A machine can run with no models, to
  show up in the cluster.

- Agent configs: `GET /api/clients/{pi,omp,opencode,claude}` downloads a
  config pointing that coding agent at every model in the cluster through the
  address the request came in on, and the Models view has download links for
  them. Same entries as ai-tools' `ai-swap-clients`, as standalone files.

- Reservations: `POST /api/lock` hands this machine to one client (e.g. a
  benchmark run) and returns a token. Until it's released (`DELETE
  /api/lock`) or lapses unrenewed (`ttl_seconds`, default 300), only
  requests carrying `X-AI-Swap-Lock: <token>` are served or may load and
  unload models here; others get 503 with Retry-After naming the holder.
  Idle unloads and resident reloads pause meanwhile, and releasing brings
  back the residents the holder unloaded. Shown in `/api/node` and the UI
  ("reserved", with a Release button); `?node=<peer>` releases a peer's.

- llama-swap compatibility: `GET /running`, `GET|POST /unload` (also
  `POST /api/unload`) and `GET /upstream/{model}/{path...}` for this
  machine's models, so tools written for llama-swap (benchmark harnesses,
  scripts that free the GPU first) work unchanged. `/api/node` and
  `/api/status` now report each running model's command line (`cmd`).

- Engine flags confirmed on an M5 Ultra (roadmap item 1): `sushi --serve` (not
  `sushi serve`) and `ds4-server` both take `--host`/`--port`, so they now get
  ports from `port_range` and two models on either engine can run side by side.
  The ds4 adapter no longer passes `--chdir`, which ds4-server doesn't have;
  `engines.ds4.dir` sets the working directory.
- New `omlx` engine ([oMLX](https://github.com/jundot/omlx), MLX with continuous
  batching). `model:` is the model's directory; the adapter serves its parent,
  sends requests under the directory name, gives each model its own SSD prompt
  cache, and loads the weights before reporting ready.
- Engines can declare a warm-up request (`Launch.WarmupPath`), POSTed after the
  health check; the model is `ready` only once it answers 2xx.
- Peers: connections time out after 3 s (a hostname that also resolves to an
  unreachable IPv6 address used to hang each new connection for 20-75 s); a
  failed forward marks the peer offline at once; a model on an offline peer
  gets `503` with `Retry-After: 10` naming the node instead of `404`.
- `/v1/models` entries carry `context_length` when the model sets `ctx`.
- Peers are dialled over IPv4 first, falling back to any address family. On
  a LAN with mDNS names, a host with IPv6 turned off made every macOS lookup
  wait 5 s for an AAAA answer (longer than a peer poll allows), and one
  publishing unreachable IPv6 addresses hung new connections.
- UI pass for several machines: a stacked memory bar per machine showing which
  models hold the memory (busy and loading ones marked), a one-line cluster
  summary, model sizes in the sidebar, a Models view with one card per machine
  (OS, version, online or last seen, memory legend, context, quiet Load
  buttons and a Chat shortcut), replies labelled with the machine that served
  them, user turns as right-aligned bubbles and flat assistant replies, a
  single rounded composer, an empty state that shows what the next message
  will do, and no empty speed tiles before the first request.
- A new logo (a brain between the swap arrows) and a small workstation icon next
  to each machine in the sidebar, red when the machine is offline.

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
