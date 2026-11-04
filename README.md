# vLLM Scheduler Playground

A self-contained browser-based demonstration of **prefill head-of-line delay and priority-related starvation** on an **unmodified vLLM server**. Two swipeable experiment cards each show four real, individually submitted chats. Each card displays first-output-token latency (TTFT) and full completion time, measured from real streaming responses. No scheduler fork, GPU instrumentation, JavaScript build tool, or modified vLLM checkout is required.

## Start in three steps

**1. Start any compatible vLLM server** with `--scheduling-policy priority`, chunked prefill, and an instruction/chat model. Example using a single GPU:

```bash
# Set this to your available physical GPU index. Do not assume GPU 3 exists.
CUDA_VISIBLE_DEVICES=0 ./start_vllm.sh
```

The helper runs `Qwen/Qwen2.5-0.5B-Instruct` with priority scheduling, chunked prefill, a 2,048-token step budget, 16 sequence slots, prefix caching disabled and a 16K context. It only *starts* ordinary installed vLLM; **it does not change vLLM code**. Knobs: `MAX_NUM_SEQS=<n>`, `MAX_MODEL_LEN=<n>` (default 16384; use 32768 with a 7B model for 16K–24K prompts), `MODEL=<id>`, `PORT=<n>`, and `MITIGATE=1`, which adds the 512-token per-prefill cap plus the adaptive floor (use it for the "after" comparison; see *Why starvation may not reproduce*). The model needs network access on first download unless already cached.

**2. Install the separate lightweight UI dependencies and launch:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_ui.sh
```

If your server is elsewhere:

```bash
VLLM_URL=http://127.0.0.1:8000 VLLM_MODEL='' ./run_ui.sh
# If vLLM requires authentication:
# VLLM_API_KEY=YOUR_KEY VLLM_URL=http://... ./run_ui.sh
```

**3. Open http://127.0.0.1:7860** in a browser on the machine running the UI.

The connected model name appears at top-right. The UI runs on port 7860, vLLM normally runs on port 8000. Both default to loopback binding; do not expose the app directly on the public internet (it has no authentication or rate limiting).

## Live demo flow

**Experiment 1: All short** — Four chats are prefilled with small questions. Click Send on each panel. They should usually receive quick responses. Chat 4 asks **“What is 2 + 2?”** at priority 2; Chats 1–3 have priority 0.

**Experiment 2: Long + short** — tick **Keep long chats busy**. The *server-side* workload keeps N real priority-0 requests (8, 12 or 16 "in flight", default 8 × ~8K-token prompts) streaming against vLLM and submits a replacement the instant one finishes. Then press **Send** on Chat 4, which asks the identical tiny question at priority 2. Chats 1–3 can still be sent manually.

The panel shows, live: background requests in flight and completed, P0 requests still awaiting a first token, vLLM's own **running / waiting queue sizes and KV usage** (from `/metrics`, a **CONTENTION** badge appears when the waiting queue is non-empty), the time left, and Chat 4's real TTFT / completion (a live "waiting" clock until its first token). The workload stops itself after **60 seconds** (hard cap), when you untick the box, press Reset, or if vLLM rejects a request (the error is shown). Stopping closes the streams, so vLLM aborts the requests.

Workload guarantees: every prompt is generated with a unique id (no prefix-cache reuse), its token length is checked with vLLM `/tokenize` (chat-templated, ±3 % of target) and again against vLLM's own `usage.prompt_tokens`; prompts for the next request are prepared while the current one runs so the measured replacement gap is ~0 ms. Starting is refused if the load would need more than 80 % of the server's KV cache or exceed its context length.

Use **Export results** to download per-request results plus the workload summary as JSON. Both experiment cards retain their replies when you swipe between them.

## What is actually measured?

- **TTFT:** milliseconds between submitting the streaming HTTP request to vLLM and receiving the first nonempty generated text chunk at the UI backend. It is not raw model forward-pass time. A waiting clock updates while no token has arrived.
- **Completion:** milliseconds until the entire streaming response has finished, measured over the same HTTP request.
- **Prompt tokens:** when the vLLM `/tokenize` endpoint is available, the actual tokenized prompt length is recorded in exported results; otherwise `null`. Long prompt target sizes are approximate, and chat templates may add tokens.
- **Priority:** sent using the standard `X-Vllm-Priority` request header. In vLLM, **lower integers are higher priority**. Nonzero priorities require `--scheduling-policy priority`.

Every chat is a real concurrent HTTP connection to the **same** vLLM model. The UI has no client-side serial queue, synthetic response delay, or scripted fake completion. Text inputs can be changed individually; Chat 4's text is synchronized between both experiment cards for an apples-to-apples comparison. Prefix caching is disabled in the included server helper; long requests also contain distinct generated text.

### Interpreting the outcome

- Three finite long requests may only produce mild HOL delay. They do **not** automatically prove indefinite starvation. Replenish high-priority arrivals for the more adversarial workload.
- A chatbot waiting a long time while other requests complete provides **user-visible evidence** of latency interference. Proving zero token *allocation* needs a scheduler trace. A 60-second run is a bounded observation, not proof of mathematically infinite waiting.
- Changes to the GPU, model, KV capacity, load level, prompt tokens, chunk cap, and max sequence slots can materially change the results. It is valid for the UI to show no severe delay on an underloaded server.
- Do not compare results from scenarios with different priorities, models, or sampling settings without documenting the change.

## Why starvation may not reproduce (diagnosis and measurements)

The earlier workload (≤ 3–4 long requests, each resubmitted only after the browser finished the previous one) did not keep vLLM's per-step token budget saturated, and the launcher's 512-token prefill cap + adaptive floor let a tiny request slip into every step. Measured on one 24 GB GPU (Qwen2.5-0.5B, 2,048-token step budget, priority policy, 8K prompts, Chat 4 sent 4 s into a 20 s run; "starved" = still waiting when the workload ended, so the true wait is at least that long):

| vLLM config | Background | Chat 4 (P2) TTFT, idle → loaded |
|---|---|---|
| 16 seqs, **512 cap + adaptive** (old default) | 8 × 8K | 81 ms → **348 ms** (no starvation) |
| same | 16 × 8K | 43 ms → **≥ 16 s, starved** |
| 16 seqs, no cap (**new default**) | 8 × 8K | 74 ms → **≥ 16 s, starved** |
| same | 12 / 16 × 8K, 16 × 12K | ≥ 16 s, starved |
| 4 seqs, no cap | 8 / 12 / 16 × 8K, 8 × 4K | ≥ 16 s, starved |

Capacity on this GPU: ~38–45 K prompt tokens/s of prefill (an 8K prompt ≈ 0.2 s of GPU time), so 8 concurrent 8K prompts already saturate the step budget; KV cache holds 1.78 M tokens and the workload peaked at 1–3 %, memory was never the limit (the ~22 GB used is vLLM's up-front reservation). More concurrency only adds waiting requests, not memory pressure. Contention was confirmed from vLLM's own gauges (waiting queue non-empty in ~95 % of 0.5 s samples).

Bottom line: starvation here is **prefill-token-budget saturation by higher-priority prompts**, not slot or memory exhaustion. Run the server with `MITIGATE=1` to compare. Results depend on GPU, model and budget; a faster GPU or smaller prompts may need higher concurrency.

## Architecture

```
Browser (vanilla JS, CSS; two swipeable experiment cards)
  └── local FastAPI UI backend (port 7860)
         └── unmodified vLLM OpenAI API (port 8000)
             ├── GET  /v1/models
             ├── GET  /metrics              (running / waiting queue sizes, KV usage)
             ├── POST /tokenize             (optional for token counts)
             └── POST /v1/chat/completions (stream=true, X-Vllm-Priority)
```

Only `fastapi`, `uvicorn`, and `httpx` are required by the UI. vLLM and its GPU dependencies remain in their existing environment. Nothing here monkey-patches or imports vLLM internals.

## Configuration

| Variable / control | Default | Meaning |
|---|---|---|
| `VLLM_URL` | `http://127.0.0.1:8000` | Root URL of vLLM (without `/v1`) |
| `VLLM_MODEL` | auto-detected | Served model ID; override if necessary |
| `VLLM_API_KEY` | empty | Optional upstream API key |
| UI `--port` | 7860 | Port for the demo UI |
| Long prompt selector | ~8K | Backend synthesizes real long text (also the background-workload size) |
| Background in-flight | 8 (8/12/16) | Concurrent P0 requests kept running |
| Keep long chats busy | off | Auto-replenishing P0 workload, hard stop at 60 s |
| Added chats | 4 per experiment | Add up to 12 chats per experiment |

Run `./run_ui.sh --host 127.0.0.1 --port 7860` to explicitly choose the UI binding. Use SSH tunneling instead of public exposure if showcasing a remote GPU server.

## Verification

```bash
python -m py_compile app.py
python -m pytest -q
```

(9 tests: priority header, streaming, unique prompts, continuous replacement, 60 s limit, capacity guards, error reporting.)

The automated tests use a local mock upstream API to validate HTTP priority forwarding, streamed output parsing, and long-text creation; **they are not GPU performance tests**. To demonstrate real scheduling behavior, run against an actual vLLM server.
