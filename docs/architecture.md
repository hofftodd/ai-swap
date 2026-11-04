# Architecture

## Goals

1. **One endpoint, many machines.** Mac (Sushi, ds4) and Linux/R9700
   (Strata, llama.cpp) today; DGX Spark and Gorgon Halo boxes later.
2. **Engine-aware.** Each engine has its own flags, port behaviour, model
   formats and platform limits. Adapters hide that.
3. **Memory-aware swapping.** These models take most of a machine's memory,
   so loading one usually means unloading another. The scheduler plans this
   instead of just reacting to out-of-memory crashes.
4. **Both client APIs.** OpenAI (`/v1/chat/completions`, ...) and Anthropic
   (`/v1/messages`) pass straight through. All four engines speak both
   natively, so ai-swap does not translate between them.
5. **A usable chat UI** shipped inside the same binary.

## Components

```
cmd/ai-swap          main: flags, wiring, graceful shutdown
internal/config      YAML schema, defaults, validation (unknown keys rejected)
internal/engine      Engine interface + adapters: sushi, ds4, strata, llamacpp, generic
internal/process     exec, process-group kill, health polling, log ring buffer
internal/scheduler   model lifecycle, memory-budget eviction, idle TTL, crash detection
internal/cluster     peer polling (/api/node) and "which node serves model X"
internal/server      HTTP: inference proxy, /v1/models, management API, UI
web/static           chat UI (single HTML file, no build step)
```

### Request flow

1. Read the body and take `model` from it.
2. **Resolve:** local model or alias that is runnable on this OS → local.
   Otherwise ask the cluster for an online peer that has it, preferring one
   where it is already `ready`. `model@node` pins a node.
3. **Local:** `scheduler.Acquire` loads the model if needed, then increments
   its in-flight count. ai-swap reverse-proxies to `127.0.0.1:<port>` with
   immediate flushing so SSE streams token by token, and releases the
   in-flight count when the response ends.
4. **Peer:** forward to the peer with `X-AI-Swap-Hop: 1`. A node that sees
   this header only serves locally, which prevents forwarding loops.

### Scheduling

All starts and stops are serialised by one lifecycle lock, so each eviction
plan sees a consistent state. To load model *M*:

- Evict any instance holding *M*'s **fixed port**. Sushi (12345), ds4 (8000)
  and Strata (8080) don't take a documented port flag, so two models on the
  same engine can't run at once unless `port_flag` is configured.
- Then evict **least-recently-used idle** models until
  `used + M.memory_gb <= memory_budget_gb`.
- If the plan needs a model that is serving a request, **wait** until it is
  released (or the client gives up). Busy models are never killed mid-stream.
- Loading is not cancelled when the client disconnects. A 100 GB model that
  is 90% loaded is kept for the retry.

A reaper unloads models idle past their `ttl`. A process that dies on its own
is marked `failed` and restarted on the next request.

### Engine adapters

| Engine | Command built | Port | Platforms | Verified? |
|---|---|---|---|---|
| `llamacpp` | `llama-server -m M -c CTX --host 127.0.0.1 --port P` | assigned | any | flags well known |
| `sushi` | `sushi serve --model DIR --ctx-size CTX` | fixed 12345 | macOS | from README; port/host flags unknown |
| `ds4` | `ds4-server -m GGUF --ctx CTX [--chdir DIR]` | fixed 8000 | macOS, Linux | from docs/SERVER.md; port flag unknown |
| `strata` | requires `cmd` | fixed 8080 or `${PORT}` | any | non-interactive server command unknown |
| `generic` | requires `cmd` | `${PORT}` or `port` | any | n/a |

Engine-level `args`, `env`, `bin`, `dir`, `port_flag` and `host_flag` let you
fix flag differences in config without changing code.

## Hardware notes

| Node | Memory | Good fits |
|---|---|---|
| Mac (Apple Silicon) | unified; budget ≈ RAM minus ~15% for macOS | Sushi, ds4 (needs 96–128 GB), MLX via `generic` |
| Linux + Radeon AI PRO R9700 | 32 GB VRAM | Strata (lists R9700 as supported), llama.cpp ROCm/Vulkan. ds4 has no R9700 build target, and DeepSeek V4 Flash wouldn't fit anyway |
| DGX Spark (future) | 128 GB unified, CUDA | ds4 (`make cuda-spark`), Strata, llama.cpp CUDA |
| Gorgon Halo (future) | up to 192 GB, up to 160 GB usable as VRAM | ds4 ROCm (Strix Halo target), Strata, llama.cpp |

## Known gaps / next steps

Roughly in priority order:

1. **Verify engine flags on real hardware.** Run `sushi serve --help`,
   `ds4-server --help`, and find Strata's non-interactive server mode. Then
   set `portFlag`/`hostFlag` in the adapters so engines get assigned ports
   and stop colliding on fixed ones.
2. **Auth.** Nodes listen on the LAN with no authentication. Add a shared
   `node.api_key` that is checked on inbound requests and sent to peers.
3. **Fairness while waiting.** A swap waiting for a busy model can starve if
   new requests keep that model busy. Fix: mark it "draining" so new
   requests queue behind the swap.
4. **Placement across nodes.** If a model is defined on two nodes, prefer the
   one where it is loaded or which has free memory, instead of local-first.
5. **Real memory readings.** Read actual usage (`vm_stat` / Metal on macOS,
   `amd-smi`/`rocm-smi` and `nvidia-smi` on Linux) and use it to check the
   configured `memory_gb` values.
6. **Model discovery.** Have adapters scan `~/.sushi/models`, GGUF folders
   and Strata's model store, and offer to create config entries.
7. **Groups.** Keep small models (embeddings, a drafter) resident next to a
   big one, like llama-swap's groups.
8. **UI.** Markdown tables and lists, image input (MiMo/GLM via Sushi and
   Strata accept images), a system prompt, parameters (temperature, max
   tokens), conversation export, and live log streaming.
9. **Packaging.** launchd plist for the Mac, systemd unit for Linux, release
   binaries for darwin/arm64, linux/amd64 and linux/arm64 (Spark).
