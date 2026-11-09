# Architecture

This document explains how ai-swap is built and why. For setup and usage,
see the [README](README.md).

## Contents

- [Goals and non-goals](#goals-and-non-goals)
- [System overview](#system-overview)
- [Packages](#packages)
- [Request lifecycle](#request-lifecycle)
- [Model lifecycle](#model-lifecycle)
- [Scheduling and eviction](#scheduling-and-eviction)
- [Engine adapters](#engine-adapters)
- [Process management](#process-management)
- [Multi-node routing](#multi-node-routing)
- [Chat UI](#chat-ui)
- [Concurrency model](#concurrency-model)
- [Design decisions](#design-decisions)
- [Hardware notes](#hardware-notes)
- [Testing](#testing)
- [Roadmap](#roadmap)

## Goals and non-goals

**Goals**

1. **One endpoint for many machines.** A Mac (Sushi, ds4) and a Linux box
   with a Radeon AI PRO R9700 (Strata, llama.cpp) today. DGX Spark and
   Gorgon Halo machines later.
2. **Engine-aware.** Each engine's flags, default port, model format and
   supported platforms are hidden behind an adapter.
3. **Memory-aware swapping.** Models that fill most of a machine's memory
   are loaded and unloaded according to a plan, rather than after an
   out-of-memory crash.
4. **Both client APIs.** Pass OpenAI and Anthropic requests through
   unchanged, streaming included.
5. **One self-contained binary**, including the chat UI. No database, no
   front-end build step.

**Non-goals (for now)**

- **Translating between APIs.** All the target engines serve both APIs
  themselves, and translating is where subtle bugs live (tool calls,
  thinking blocks, cache control).
- **Doing inference.** ai-swap only starts processes and proxies requests.
- **Central coordination.** There is no leader. Every machine is equal and
  owns its own models.
- **Load-balancing replicas** of one model across machines.

## System overview

```
  clients (chat UI, OpenAI SDKs, Claude Code)
     │
     ▼
  ┌─ node "mac" ───────────────────────────────────────────────────┐
  │ server ─ resolve(model)                                        │
  │   ├─ local ─► scheduler.Acquire ─► process ─► sushi  :12345    │
  │   │             (evict LRU idle,           ─► ds4    :8000     │
  │   │              wait for busy)                                │
  │   └─ remote ─► cluster.Find ─► reverse proxy ──────────┐       │
  └────────────────────────────────────────────────────────┼───────┘
  X-AI-Swap-Hop: 1                                         │
  ┌─ node "linux" ─────────────────────────────────────────▼───────┐
  │ server ─► scheduler ─► process ─► strata / llama-server :91xx  │
  └────────────────────────────────────────────────────────────────┘

  Every node polls each peer's /api/node every 10 s.
```

Every machine runs the same binary. A machine owns the engines it starts
and knows about other machines only through what they report at
`/api/node`.

## Packages

| Package | Responsibility | Depends on |
|---|---|---|
| `cmd/ai-swap` | Flags, wiring, signal handling, `-check` | everything |
| `internal/config` | YAML schema, defaults, validation, `~` expansion, name/alias lookup | `yaml.v3` |
| `internal/engine` | `Engine` interface, built-in adapters, command splitting | `config` |
| `internal/process` | Start a process in its own process group, health polling, stop, log ring buffer | `engine` |
| `internal/scheduler` | Per-model state machine, eviction planning, TTL reaper, crash detection | `config`, `engine`, `process` |
| `internal/cluster` | Peer polling, peer state, choosing a peer for a model | `config`, `scheduler` (status types) |
| `internal/server` | HTTP routing, reverse proxy, `/v1/models` merging, management API, error formats | all of the above |
| `internal/version` | Release version (overridable with `-ldflags -X`) | – |
| `web` | Embedded `static/index.html` | – |

Dependencies only point downward. The scheduler knows nothing about HTTP,
and the server never touches processes directly.

## Request lifecycle

For `POST /v1/chat/completions` (or any other inference path):

1. **Read the body.** Up to 64 MB is read into memory, because it has to be
   parsed for `model` and may be rewritten.
2. **Parse `model`.** Missing → 400. Not JSON → 400.
3. **Resolve the target** (`server.resolve`):
   - Split `name@node`.
   - If no node was given, or it is this machine: look up the name or alias
     in this machine's config. A match whose engine can run on this OS
     (`Engine.Supported()`) is served locally.
   - Otherwise, unless the request was already forwarded by a peer, ask
     `cluster.Find`.
   - Nothing found → 404 in the client's API error format.
4. **Serve locally:**
   - `scheduler.Acquire(ctx, model)` blocks until the model is `ready`
     (loading it if needed), then increments `inflight` and returns the
     port and a `release` function.
   - Rewrite `model` in the body if `upstream_model` is set, or if the
     client used an alias.
   - Reverse-proxy to `127.0.0.1:<port>` with `FlushInterval: -1`, so each
     SSE chunk is passed on as soon as it arrives.
   - `release()` runs when the response finishes or the client disconnects.
5. **Forward to a peer:** rewrite `model` to the peer's real model name (to
   resolve aliases and drop `@node`), add `X-AI-Swap-Hop: 1`, and
   reverse-proxy to the peer.

The shared transport sets no response-header timeout: prefill on a 100k-token
prompt can take minutes before the first byte.

## Model lifecycle

```
            Load()                    health OK
 stopped ───────────► starting ───────────────────► ready ◄──┐
    ▲                    │                          │  │      │ Acquire/release
    │                    │ start error /            │  └──────┘
    │                    │ health timeout /         │
    │                    ▼ exits early              │ process exits by itself
    │                 failed ◄──────────────────────┘
    │                    │
    │                    │ next Acquire() retries the start
    │                    ▼
    │                 starting ...
    │
    │   stop()  (evict / unload / TTL / shutdown)
    └─────────── stopping ◄── ready
```

- **`starting`, `ready` and `stopping` count against the memory budget.**
  `failed` and `stopped` don't.
- A `failed` model keeps its last error (shown in the UI). The next request
  tries again.
- A crash is detected by a goroutine watching the process. It only marks
  the model `failed` if it was still `ready` with the same process, so an
  intentional stop is never reported as a crash.

## Scheduling and eviction

Every start and stop goes through **one lifecycle lock** (a channel of
capacity 1, so waiting on it can be cancelled). Each plan is therefore
computed against a settled state, never a half-finished swap.

To load model **M**, `Manager.Load` repeats until M is ready:

1. **Plan** (`plan()`, a pure function that is tested on its own):
   1. Any active model on M's **fixed port** must go. If one of them is
      busy, wait.
   2. With the remaining active models, if `used + M.memory_gb > budget`,
      evict idle models **oldest last-use first** until it fits.
   3. If it still doesn't fit, the rest are busy, so wait.
2. **Wait** (if needed) on a "something changed" channel, which is closed
   whenever any model changes state or a request finishes, or until the
   client's context is cancelled. Then plan again.
3. **Evict** the chosen models (SIGTERM, then SIGKILL after 15 s), then plan
   again. Planning again is cheap and copes with anything that changed in
   between.
4. **Start** M: build the launch spec, start the process, and poll its
   health endpoint until `health_timeout`.

**Rules this keeps:**

- A model serving a request is never evicted by a swap. (An explicit
  `unload` does stop it.)
- Two models on the same fixed port never run at once.
- If the budget is set, the total `memory_gb` of active models never exceeds
  it. Config validation rejects a model larger than the budget, so a plan
  can always succeed eventually.

**Loading ignores the client's cancellation.** If a client gives up after
60 s on a 100 GB model that needs 90 s, the load still finishes, and the
retry gets a ready model. The client's cancellation only stops *waiting*.

**TTL reaper.** Every 15 s it unloads `ready`, idle models whose `ttl` has
passed. It only tries the lifecycle lock and skips that round if a swap is
in progress, so it never holds up a request.

**Known weakness: starvation.** If a swap is waiting for model A to become
idle and new requests keep arriving for A, the swap can wait indefinitely.
The planned fix is a "draining" flag: once a pending plan needs A gone, new
requests for A queue behind the swap.

## Engine adapters

```go
type Engine interface {
    Name() string
    Supported() bool   // can this engine run on runtime.GOOS?
    Caps() Caps        // {OpenAI, Anthropic}
    Build(m *config.Model, ec config.EngineConfig, allocPort func() (int, error)) (*Launch, error)
}

type Launch struct {
    Argv       []string
    Dir        string
    Env        []string
    Port       int
    HealthPath string
}
```

The built-in adapters are table-driven (`builtin` in `engine.go`), because
engines mostly differ only in flag names and port behaviour.

**How the port is chosen** (first match wins):

1. `models.X.port` is set → use it.
2. The engine has a port flag (built in, or `engines.E.port_flag`), or `cmd`
   contains `${PORT}` → take a free port from `node.port_range`.
3. The engine has a fixed default port → use it. The scheduler then treats
   two models on that engine as conflicting.
4. Otherwise → config error.

**How the command line is built:** `bin [subcmd] [modelFlag model]
[ctxFlag ctx] [hostFlag 127.0.0.1] [portFlag port] [engine extras]
[engines.E.args] [models.X.args]`, or `cmd` with placeholders filled in.
`cmd` is split into arguments without a shell, so model paths can't inject
shell commands.

| Adapter | Binary | Model / ctx flags | Port | Health | OS | Source of truth |
|---|---|---|---|---|---|---|
| `llamacpp` | `llama-server` | `-m` / `-c` | `--port` (assigned), `--host` | `/health` | any | llama.cpp docs |
| `sushi` | `sushi --serve` | `--model` / `--ctx-size` | `--port` (assigned), `--host` | `/v1/models` | darwin | `sushi --help` (1.4.1) |
| `ds4` | `ds4-server` | `-m` / `--ctx` | `--port` (assigned), `--host` | `/v1/models` | darwin, linux | `ds4-server --help` |
| `omlx` | `omlx serve` | `--model-dir` = parent of `model` | `--port` (assigned), `--host` | `/v1/models`, then POST `/v1/models/{id}/load` | darwin | `omlx serve --help` (0.7.3) |
| `strata` | (needs `cmd`) | – | fixed 8080 or `${PORT}` | `/v1/models` | any | Strata README |
| `generic` | (needs `cmd`) | – | `${PORT}` or `port` | `/v1/models` | any | – |

**Adding an engine:** add a `register(&builtin{...})` entry in `engine.go`
and a case in `engine_test.go`. If an engine needs logic that a table can't
express (for example, finding models on disk), implement `Engine` directly.

## Process management

- **Process groups.** Each engine starts with `Setpgid`, and stopping sends
  the signal to the whole group (`kill(-pgid)`). Wrapper scripts and their
  children (such as `strata launch` or `sh -c ...`) are stopped too, so
  nothing is left holding a port or GPU memory. `process_test.go` checks
  this by confirming the port is free after stopping a `sh` → `python` pair.
- **Stopping:** SIGTERM, wait up to 15 s, then SIGKILL.
- **Health check:** GET `http://127.0.0.1:<port><health_path>` every 500 ms
  until a 200. Fails early if the process exits first.
- **Warm-up:** if the adapter sets a warm-up path (oMLX), it is POSTed once
  the health check passes, and the model is `ready` only when that answers
  2xx. oMLX answers health checks before loading any weights, so without
  this "ready" would mean "listening" and the first request would pay for
  the load.
- **Logs:** stdout and stderr go to a 256 KB ring buffer per model, prefixed
  with ai-swap's own `exec` and `exited` lines. They are shown at
  `/api/models/{name}/logs` and in the UI.
- **Shutdown:** on SIGINT/SIGTERM, ai-swap stops the HTTP server (5 s grace),
  then stops every model in parallel.

## Multi-node routing

- **Discovery** is static: `peers` in config. Each machine polls
  `GET <peer>/api/node` every 10 s (5 s timeout). The answer contains the
  peer's name, version, OS/arch, memory budget and use, and a status for
  every model.
- **Fresh state after forwarding:** after forwarding an inference or
  load/unload request, the machine re-polls that peer straight away
  (`cluster.Refresh`), so a model the peer just loaded shows as `ready`
  without waiting up to 10 s.
- **Failing fast:** peer connections time out after 3 s. A forward that
  can't connect marks the peer offline immediately, and a request for a
  model that only an offline peer has gets `503` + `Retry-After`, naming the
  node, rather than `404`. (Offline peers keep their last model list for
  this.) The 3 s limit matters with mDNS names: if a hostname also resolves
  to an IPv6 address that a firewall silently drops, every new connection
  would otherwise wait 20-75 s.
- **Choosing a peer** (`cluster.Find`): online peers that have the name (or
  alias) and can run it, preferring one where it is already `ready`.
  `model@node` matches either the peer's configured name or the name it
  reports for itself.
- **Loop prevention:** forwarded requests (including the polls) carry
  `X-AI-Swap-Hop: 1`. A machine that sees it only serves its own models and
  only lists its own models in `/v1/models`.
- **Duplicate names:** `/v1/models` lists the local model under its plain
  name and the peer's copy as `name@node`, so both can be addressed.
- **Same config everywhere:** an engine that can't run on this OS (for
  example `sushi` on Linux) is skipped locally and routed to a peer. In
  principle one shared model list can be used on every machine; only `node`
  and `peers` need to differ.

## Chat UI

`web/static/index.html` is a single file with no framework and no build
step, embedded with `go:embed`.

- Polls `/v1/models` and `/api/status` every 3 s.
- Sends `/v1/chat/completions` with `stream: true` and
  `stream_options.include_usage`, and parses the SSE stream itself.
  `delta.reasoning_content` / `delta.reasoning` go into a collapsible
  "Thinking" section.
- Shows time to first token, tokens/sec (from `usage` when the engine sends
  it, otherwise by counting chunks) and token counts.
- Conversations live in `localStorage`. They are per browser and never sent
  to the server except as the messages of a request.
- Light and dark themes follow the OS setting.

## Reservations

`lock.go`. A reservation is one token per machine, held in the server (not
the scheduler), with an owner, a reason and an expiry. Every local entry point
(inference, load/unload, `/unload`, `/upstream`) passes through `lockGate`:
without a reservation everything is allowed; with one, only requests carrying
the token are, and each of those renews the expiry and counts as in flight,
so a single 20-minute request can't let the reservation lapse under it.
Taking a reservation pauses the scheduler's reaper (idle unloads, resident
reloads); releasing it, or its lapsing, resumes them and clears the
"unloaded by hand" mark on residents, so they come back. The reservation is
published in `/api/node`, so peers, the UI and ai-factory can see it.

## Concurrency model

| Shared state | Protected by |
|---|---|
| Instance fields (`state`, `handle`, `port`, `inflight`, `lastUsed`, `lastErr`) | `Manager.mu` (held briefly, never during I/O) |
| Start/stop ordering | `Manager.lifecycle` (1-slot channel, held during I/O) |
| Waking waiters | `Manager.changed`: closed and replaced under `mu` on every change |
| Peer table | `Cluster.mu` (RWMutex) |
| Log buffers | `RingBuffer.mu` |

Lock order is always `lifecycle` → `mu`. Code never holds `mu` while
taking `lifecycle`. Tests run with `-race`.

## Design decisions

**New program instead of a llama-swap fork.** llama-swap's config and
process model assume one machine and command lines written by hand. The
features this project needs (adapters, a memory budget, peers) touch most
of it. Re-implementing the useful core (swap on demand, proxy, TTL) took
less code than a fork would have had to change.

**Every machine is the same; no coordinator.** It's simpler to run (one
binary, one config per machine), any machine can be the front door, and
losing one machine only loses its models.

**Configured `memory_gb` instead of measured memory.** It's predictable and
works the same on Metal, ROCm and CUDA. Measuring is on the roadmap as a
cross-check, not as a replacement.

**Proxy the raw request instead of translating.** All target engines speak
both APIs, so passing bytes through is faster and avoids subtle
incompatibilities. Only the `model` field is ever rewritten.

**Read the whole body before proxying.** It's needed to route by `model`.
Bodies are capped at 64 MB, which leaves room for long contexts and
base64-encoded images.

**Unix only.** Process groups make cleanup reliable. Windows support would
need Job Objects instead.

## Hardware notes

| Machine | Memory | Good fits | Notes |
|---|---|---|---|
| Mac (Apple Silicon) | Unified. Budget ≈ RAM minus ~15% | Sushi, ds4 (96–128 GB), MLX via `generic` | Sushi's docs warn that over-committing wired memory can kernel-panic macOS |
| Linux + Radeon AI PRO R9700 | 32 GB VRAM | Strata (lists R9700 as supported), llama.cpp (ROCm or Vulkan) | ds4 has no R9700 build target, and DeepSeek V4 Flash wouldn't fit anyway |
| DGX Spark (future) | 128 GB unified, CUDA, arm64 | ds4 (`make cuda-spark`), Strata, llama.cpp | Cross-compile with `GOARCH=arm64` |
| Gorgon Halo / Ryzen AI Max 400 (future) | Up to 192 GB, up to 160 GB usable as VRAM | ds4 (Strix Halo ROCm target), Strata, llama.cpp | Memory bandwidth, not capacity, will limit speed |

## Testing

| Package | What is covered |
|---|---|
| `config` | Defaults, TTL handling, alias resolution, validation errors, unknown-key rejection, example configs parse |
| `engine` | Exact command line per adapter, port selection rules, `${PORT}` handling, command splitting edge cases |
| `process` | Real process: start, health, stopping the whole group frees the port, early exit, ring buffer |
| `scheduler` | LRU eviction, waiting for busy models, fixed-port conflicts, health failure, TTL reaper, crash detection and restart (using a fake process starter) |
| `server` | Two real in-process machines with fake backends: local routing and `upstream_model`, peer forwarding with aliases and `@node`, merged `/v1/models`, SSE pass-through, error formats, load/unload forwarded to a peer |

```bash
go test -race ./...
```

## Roadmap

Roughly in priority order:

1. **Strata's non-interactive server command.** (Sushi's and ds4's flags
   were confirmed on an M5 Ultra; both now get assigned ports.)
2. **Authentication.** A shared `node.api_key`, checked on inbound requests
   (Bearer or `x-api-key`) and sent to peers.
3. **Fair swapping.** Add the "draining" flag described under
   [Scheduling](#scheduling-and-eviction).
4. **Smarter placement.** When a model exists on several machines, prefer
   one where it is loaded, or which has free memory, over local-first.
5. **Measured memory.** Read actual use with `vm_stat` (macOS) and
   `amd-smi`/`rocm-smi`/`nvidia-smi` (Linux), and compare it with
   `memory_gb`.
6. **Model discovery.** Adapters scan `~/.sushi/models`, GGUF folders and
   Strata's model store, and offer to create config entries.
7. **Groups.** Keep small models (embeddings, drafters) loaded next to a
   large one.
8. **UI.** Full Markdown (lists, tables), image input, system prompt,
   sampling parameters, export, and live log streaming.
9. **Packaging.** (A launchd plist is in `examples/`.) Systemd unit, release binaries for
   darwin/arm64, linux/amd64 and linux/arm64, and config hot-reload.
