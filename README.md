# ai-swap

One front door for local models spread across your machines. It is in the
spirit of [llama-swap](https://github.com/mostlygeek/llama-swap), but built
for the newer single-model engines:

- **[Sushi](https://github.com/joshuaswarren/sushi)**: Apple Silicon, custom quants, SSD expert streaming
- **[ds4 / DwarfStar4](https://github.com/antirez/ds4)**: DeepSeek V4 Flash on Metal, CUDA (DGX Spark) and ROCm (Strix Halo)
- **[Strata](https://github.com/Niko1221/Strata)**: large MoE models on consumer GPUs, including the Radeon AI PRO R9700
- **llama.cpp** (`llama-server`) and any other OpenAI-compatible server via `generic`

You send an OpenAI or Anthropic request to any node. ai-swap starts the right
engine on the right machine, evicting idle models when memory is short, and
streams the answer back. A built-in chat UI shows every node's models and
memory.

```
                 ┌──────────── mac (Apple Silicon) ─────────────┐
 clients ───────►│ ai-swap :8686 ──► sushi :12345 / ds4 :8000    │
 (chat UI,       │      │                                        │
  Claude Code,   └──────┼────────────────────────────────────────┘
  OpenAI SDKs)          │ peers: model not here? forward it
                 ┌──────▼──── linux (R9700, 32 GB) ─────────────┐
                 │ ai-swap :8686 ──► strata / llama-server :91xx │
                 └───────────────────────────────────────────────┘
```

Every machine runs the same binary with its own config. Each node lists the
others as `peers`, so you can point clients at whichever node is convenient.

## Quick start

```bash
go build -o ai-swap ./cmd/ai-swap
```

```bash
./ai-swap -config examples/mac.yaml -check   # validate + print launch commands
```

```bash
./ai-swap -config examples/mac.yaml
```

Open `http://<node>:8686/` for the chat UI, or point a client at it:

```bash
curl http://mac.local:8686/v1/chat/completions \
  -d '{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

For Claude Code or other Anthropic-API clients, set
`ANTHROPIC_BASE_URL=http://mac.local:8686`. The Anthropic endpoint
`/v1/messages` is routed the same way.

## Config

See [`examples/mac.yaml`](examples/mac.yaml) and
[`examples/linux-r9700.yaml`](examples/linux-r9700.yaml). The main fields:

| Field | Meaning |
|---|---|
| `node.memory_budget_gb` | How much (V)RAM loaded models may use together. Models are evicted least-recently-used to stay under it. |
| `peers[]` | Other ai-swap nodes. Requests for models this node doesn't have (or can't run on this OS) are forwarded. |
| `models.<name>.engine` | `sushi`, `ds4`, `strata`, `llamacpp`, or `generic` |
| `models.<name>.memory_gb` | Expected footprint while loaded |
| `models.<name>.ttl` | Unload after this much idle time (`0` = never) |
| `models.<name>.cmd` | Full command line override; `${PORT}`, `${MODEL}`, `${CTX}` are substituted |
| `models.<name>.aliases` | Extra names clients can use |
| `models.<name>.upstream_model` | Rewrites the `model` field sent to the backend |
| `engines.<engine>.port_flag` | Teach ai-swap an engine's port flag so it can assign ports itself |

Clients can pin a node with `model@node`, for example `coder@linux`.

## API

| Endpoint | |
|---|---|
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings` | OpenAI-compatible, routed by `model` |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic-compatible, routed by `model` |
| `GET /v1/models` | Models on this node and all online peers |
| `GET /api/status` | Node, memory and model states, peers |
| `POST /api/models/{name}/load` / `unload` | Manual control (`name@node` works) |
| `GET /api/models/{name}/logs` | Last 256 KB of the engine's output |

## Status

This is an early skeleton. Engine flags for Sushi and Strata are not yet
verified on real hardware; see [docs/architecture.md](docs/architecture.md)
for the design, known gaps and roadmap.

```bash
go test -race ./...
```
