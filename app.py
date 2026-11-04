"""Standalone presentation UI for an unmodified OpenAI-compatible vLLM server."""
import argparse
import asyncio
import json
import os
import random
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

MAX_PROMPT_TOKENS = 30000   # needs a server started with a context window to match (32K for Qwen2.5-7B)
ROOT = Path(__file__).parent
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_MODEL = os.getenv("VLLM_MODEL", "")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")


@asynccontextmanager
async def lifespan(_app):
    yield
    # Shutdown: stop any background workload so no requests are left hammering vLLM.
    if WORKLOAD:
        WORKLOAD.stop("server shutdown")
        if WORKLOAD_TASK:
            await asyncio.wait({WORKLOAD_TASK}, timeout=5)


app = FastAPI(title="vLLM Scheduler Playground", docs_url=None, redoc_url=None, lifespan=lifespan)


class ChatRequest(BaseModel):
    # Short prompts and long-prompt *instructions* are capped at 2,000 chars; an `exact` prompt is a
    # fully generated long document (shown in the UI) that is sent to vLLM as-is.
    prompt: str = Field(min_length=1, max_length=200_000)
    exact: bool = False
    kind: Literal["short", "long"] = "short"
    target_tokens: int = Field(default=4096, ge=512, le=MAX_PROMPT_TOKENS)
    priority: int = Field(default=0, ge=-100, le=100)
    max_output_tokens: int = Field(default=32, ge=1, le=128)
    # Unique request prefix avoids accidental full-prefix cache reuse.
    seed: str = Field(default="", max_length=100)

    @model_validator(mode="after")
    def _cap_non_exact(self):
        if not self.exact and len(self.prompt) > 2000:
            raise ValueError("prompt longer than 2,000 characters must be sent with exact=true")
        return self


def upstream_headers(priority=None):
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    if priority is not None:
        headers["X-Vllm-Priority"] = str(priority)
    return headers


async def model_name(client):
    if VLLM_MODEL:
        return VLLM_MODEL
    response = await client.get(f"{VLLM_URL}/v1/models", headers=upstream_headers(), timeout=5)
    response.raise_for_status()
    models = response.json().get("data", [])
    if not models:
        raise RuntimeError("vLLM returned no served models")
    return models[0]["id"]


def compose_long_prompt(instruction, seed, sections):
    # Variation near the beginning also reduces accidental shared-prefix caching.
    rng = random.Random(seed)
    topics = ("scheduler", "token budget", "queueing", "latency", "memory", "throughput",
              "batching", "prefill", "cache locality", "fairness", "request arrival", "GPU")
    parts = [f"Independent technical report {seed}. Read the full report and then: {instruction}\n\n"]
    for i in range(sections):
        topic = topics[rng.randrange(len(topics))]
        metric = rng.randrange(1000, 9999)
        parts.append(
            f"Section {i + 1}, sample {metric}: The {topic} experiment measures response times "
            f"under concurrent requests. Measurements are compared across workloads and "
            f"the observation is recorded separately for each test. The scheduling trace "
            f"contains enough context to support a reproducible analysis.\n"
        )
    return "".join(parts)


async def tokenize_count(client, model, prompt):
    try:
        r = await client.post(f"{VLLM_URL}/tokenize", headers=upstream_headers(),
                              json={"model": model, "prompt": prompt}, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data.get("count"), int):
            return data["count"]
        if isinstance(data.get("tokens"), list):
            return len(data["tokens"])
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        pass
    return None


async def prepare_prompt(client, model, req):
    if req.exact:     # the user is looking at this exact text in the UI: send it untouched
        return req.prompt, await tokenize_count(client, model, req.prompt)
    if req.kind == "short":
        return req.prompt.strip(), await tokenize_count(client, model, req.prompt.strip())
    # Produce *real* long text, not a short instruction merely labeled '4K'.
    seed = req.seed or uuid.uuid4().hex
    sections = max(10, req.target_tokens // 42)
    prompt = compose_long_prompt(req.prompt, seed, sections)
    count = await tokenize_count(client, model, prompt)
    # Adjust once or twice if the optional /tokenize endpoint is available.
    for _ in range(3):
        if not count or abs(count - req.target_tokens) <= max(48, req.target_tokens // 40):
            break
        sections = max(1, int(sections * (req.target_tokens / count)))
        prompt = compose_long_prompt(req.prompt, seed, sections)
        count = await tokenize_count(client, model, prompt)
    return prompt, count


def ndjson(data):
    return json.dumps(data, ensure_ascii=False) + "\n"


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/app.js")
async def script():
    return FileResponse(ROOT / "static" / "app.js", media_type="application/javascript")


@app.get("/styles.css")
async def styles():
    return FileResponse(ROOT / "static" / "styles.css", media_type="text/css")


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            model = await model_name(client)
        return {"online": True, "model": model, "url": VLLM_URL}
    except (httpx.HTTPError, ValueError, KeyError, RuntimeError) as exc:
        return {"online": False, "message": str(exc), "url": VLLM_URL}


class LongPromptRequest(BaseModel):
    prompt: str = Field(default="Summarize this large technical report.", min_length=1, max_length=2000)
    target_tokens: int = Field(default=8192, ge=512, le=MAX_PROMPT_TOKENS)


@app.post("/api/long-prompt")
async def long_prompt(req: LongPromptRequest):
    """Generate a real, unique long document (token length verified by vLLM) for display in a chat box."""
    async with httpx.AsyncClient() as client:
        try:
            model = await model_name(client)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            raise HTTPException(503, f"Cannot reach vLLM at {VLLM_URL}: {exc}")
        chat_req = ChatRequest(prompt=req.prompt, kind="long", target_tokens=req.target_tokens,
                               seed=uuid.uuid4().hex)
        text, count = await prepare_prompt(client, model, chat_req)
    return {"prompt": text, "prompt_tokens": count, "target_tokens": req.target_tokens}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def send():
        # This endpoint is deliberately stateless: all concurrency is controlled
        # by actual simultaneous HTTP connections to the same vLLM server.
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
            try:
                model = await model_name(client)
                prompt, prompt_tokens = await prepare_prompt(client, model, req)
                yield ndjson({"event": "prepared", "prompt_tokens": prompt_tokens,
                                  "prompt_chars": len(prompt), "priority": req.priority})
                payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                           "stream": True, "max_tokens": req.max_output_tokens,
                           "temperature": 0,
                           # Ask vLLM for the authoritative prompt-token count (final usage chunk).
                           "stream_options": {"include_usage": True}}
                start = time.perf_counter()
                yield ndjson({"event": "started", "priority_header": str(req.priority)})
                first_ms = None
                output = 0
                server_tokens = None
                async with client.stream("POST", f"{VLLM_URL}/v1/chat/completions",
                                         headers=upstream_headers(req.priority), json=payload) as r:
                    if r.status_code != 200:
                        detail = (await r.aread()).decode("utf-8", errors="replace")[:600]
                        yield ndjson({"event": "error", "message": f"vLLM HTTP {r.status_code}: {detail}"})
                        return
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            packet = json.loads(raw)
                            usage = packet.get("usage")
                            if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
                                server_tokens = usage["prompt_tokens"]
                            choice = (packet.get("choices") or [{}])[0]
                            delta = choice.get("delta") or {}
                            content = delta.get("content") or ""
                        except (KeyError, TypeError, IndexError, json.JSONDecodeError):
                            continue
                        if not isinstance(content, str) or not content:
                            continue
                        now = time.perf_counter()
                        if first_ms is None:
                            first_ms = round((now - start) * 1000, 2)
                        output += len(content)
                        yield ndjson({"event": "token", "text": content, "ttft_ms": first_ms})
                yield ndjson({"event": "done", "ttft_ms": first_ms,
                                  "total_ms": round((time.perf_counter() - start) * 1000, 2),
                                  "output_chars": output, "prompt_tokens": server_tokens or prompt_tokens,
                                  "server_prompt_tokens": server_tokens})
            except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as exc:
                yield ndjson({"event": "error", "message": str(exc)})
            except asyncio.CancelledError:
                raise
    return StreamingResponse(send(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Background priority-0 prefill workload (Experiment 2)
# ---------------------------------------------------------------------------
# A server-side set of N workers. Each worker keeps exactly one real, streaming
# priority-0 request in flight against vLLM; the moment it completes, the next
# (already prepared, token-verified, unique) prompt is submitted. Nothing here
# simulates latency: every number comes from real HTTP streams or vLLM /metrics.

MAX_WORKLOAD_SECONDS = 60
MAX_CONCURRENCY = 16
TOKEN_TOLERANCE = 0.03          # accepted |actual - target| / target
KV_SAFETY_FRACTION = 0.80       # refuse workloads needing more than this share of KV cache
WORKLOAD_INSTRUCTION = "Summarize the report above in one short sentence."
METRIC_NAMES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "preemptions": "vllm:num_preemptions_total",
}


class WorkloadError(RuntimeError):
    """A meaningful, user-presentable workload failure."""


class WorkloadRequest(BaseModel):
    concurrency: int = Field(default=8, ge=1, le=MAX_CONCURRENCY)
    target_tokens: int = Field(default=8192, ge=1024, le=MAX_PROMPT_TOKENS)
    max_output_tokens: int = Field(default=16, ge=1, le=64)
    duration_s: float = Field(default=MAX_WORKLOAD_SECONDS, ge=5, le=MAX_WORKLOAD_SECONDS)


def parse_metrics(text):
    """Pull the few scheduler gauges we need out of Prometheus text."""
    out = {key: None for key in METRIC_NAMES}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key, name in METRIC_NAMES.items():
            if line.startswith(name) and line[len(name):len(name) + 1] in ("{", " "):
                try:
                    value = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    continue
                out[key] = (out[key] or 0.0) + value
        if line.startswith("vllm:cache_config_info"):
            blocks = re.search(r'num_gpu_blocks="(\d+)"', line)
            size = re.search(r'block_size="(\d+)"', line)
            if blocks and size:
                out["kv_capacity_tokens"] = int(blocks.group(1)) * int(size.group(1))
    return out


_sched_cache = {"at": 0.0, "value": {"available": False}}


async def fetch_scheduler(client=None, max_age=0.3):
    """Latest vLLM scheduler gauges (running / waiting queue sizes, KV usage)."""
    now = time.monotonic()
    if now - _sched_cache["at"] < max_age:
        return _sched_cache["value"]
    own = client is None
    client = client or httpx.AsyncClient()
    try:
        r = await client.get(f"{VLLM_URL}/metrics", headers=upstream_headers(), timeout=3)
        r.raise_for_status()
        value = parse_metrics(r.text)
        value["available"] = value.get("waiting") is not None
    except (httpx.HTTPError, ValueError):
        value = {"available": False}
    finally:
        if own:
            await client.aclose()
    _sched_cache.update(at=now, value=value)
    return value


async def tokenize_chat(client, model, prompt):
    """Exact token count of the chat-templated request, via vLLM's /tokenize."""
    try:
        r = await client.post(f"{VLLM_URL}/tokenize", headers=upstream_headers(), timeout=20,
                              json={"model": model, "messages": [{"role": "user", "content": prompt}]})
    except httpx.HTTPError as exc:
        raise WorkloadError(f"cannot reach vLLM /tokenize to verify prompt length: {exc}")
    if r.status_code != 200:
        raise WorkloadError(f"vLLM /tokenize returned HTTP {r.status_code}; prompt length cannot be "
                            f"verified, so the workload was not started")
    count = r.json().get("count")
    if not isinstance(count, int):
        raise WorkloadError("vLLM /tokenize returned no token count; cannot verify prompt length")
    return count


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)


class Workload:
    def __init__(self, req, model, limits):
        self.req, self.model, self.limits = req, model, limits
        self.state = "running"            # running | stopping | finished | error
        self.reason = ""
        self.message = ""
        self.started = time.monotonic()
        self.finished = None
        self.in_flight = 0
        self.awaiting = 0                 # P0 requests submitted but no first token yet
        self.submitted = 0
        self.completed = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.last_error = None
        self.verified = 0
        self.token_mismatch = 0
        self.est_tokens = []              # /tokenize count before submit
        self.server_tokens = []           # usage.prompt_tokens reported by vLLM
        self.ttfts, self.totals, self.gaps = [], [], []
        self.completed_tokens = 0
        self.max_waiting = 0
        self.waiting_samples = 0
        self.samples = 0
        self.max_running = 0
        self.max_kv = 0.0
        self.sections_per_token = None
        self.tasks = []
        self._stopping = False

    # -- lifecycle ---------------------------------------------------------
    def active(self):
        return self.state == "running"

    async def run(self):
        limits = httpx.Limits(max_connections=self.req.concurrency * 2 + 8)
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10), limits=limits) as client:
            self.tasks = [asyncio.create_task(self.worker(client)) for _ in range(self.req.concurrency)]
            self.tasks.append(asyncio.create_task(self.poll_scheduler(client)))
            try:
                await asyncio.wait_for(self._until_stopped(), timeout=self.req.duration_s)
                self.reason = self.reason or "stopped"
            except asyncio.TimeoutError:
                self.reason = "time limit reached"
            except asyncio.CancelledError:
                self.reason = self.reason or "cancelled"
            finally:
                await self._shutdown()

    async def _until_stopped(self):
        while self.active():
            await asyncio.sleep(0.1)

    async def _shutdown(self):
        if self.state == "running":
            self.state = "stopping"
        for t in self.tasks:
            t.cancel()                    # closes open streams -> vLLM aborts the requests
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.in_flight = self.awaiting = 0
        if self.state != "error":
            self.state = "finished"
        self.finished = time.monotonic()

    def stop(self, reason="stopped by user"):
        if self.active():
            self.reason = reason
            self.state = "stopping"

    def fail(self, message):
        if self.state in ("running", "stopping"):
            self.message = message
            self.reason = "error"
            self.state = "error"

    # -- prompt preparation (off the critical path: runs while a request is in flight) -----
    async def prepare(self, client):
        target = self.req.target_tokens
        seed = uuid.uuid4().hex                       # unique per prompt -> no prefix-cache reuse
        sections = self.sections_per_token and max(1, round(target * self.sections_per_token)) \
            or max(10, target // 42)
        count = None
        for _ in range(5):
            prompt = compose_long_prompt(WORKLOAD_INSTRUCTION, seed, sections)
            count = await tokenize_chat(client, self.model, prompt)
            if abs(count - target) <= target * TOKEN_TOLERANCE:
                self.sections_per_token = sections / count
                self.verified += 1
                return prompt, count
            sections = max(1, round(sections * target / count))
        raise WorkloadError(f"could not generate a {target}-token prompt within "
                            f"{TOKEN_TOLERANCE:.0%} (last attempt: {count} tokens)")

    async def worker(self, client):
        nxt = asyncio.create_task(self.prepare(client))
        last_done = None
        try:
            while self.active():
                try:
                    prompt, count = await nxt
                except WorkloadError as exc:
                    self.fail(str(exc))
                    return
                if not self.active():
                    return
                nxt = asyncio.create_task(self.prepare(client))   # prefetch the replacement
                if last_done is not None:
                    self.gaps.append((time.perf_counter() - last_done) * 1000)
                await self.one_request(client, prompt, count)
                last_done = time.perf_counter()
        finally:
            nxt.cancel()

    # -- one real streamed request ---------------------------------------
    async def one_request(self, client, prompt, est_tokens):
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "stream": True, "max_tokens": self.req.max_output_tokens, "temperature": 0,
                   "stream_options": {"include_usage": True}}
        self.submitted += 1
        self.in_flight += 1
        self.awaiting += 1
        got_first = False
        server_tokens = None
        start = time.perf_counter()
        ttft = None
        try:
            async with client.stream("POST", f"{VLLM_URL}/v1/chat/completions",
                                     headers=upstream_headers(0), json=payload) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode("utf-8", errors="replace")[:300]
                    raise WorkloadError(f"vLLM HTTP {r.status_code}: {detail}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        packet = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    usage = packet.get("usage")
                    if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
                        server_tokens = usage["prompt_tokens"]
                    delta = ((packet.get("choices") or [{}])[0].get("delta") or {})
                    if not got_first and isinstance(delta.get("content"), str) and delta["content"]:
                        got_first = True
                        ttft = (time.perf_counter() - start) * 1000
                        self.awaiting -= 1
            total = (time.perf_counter() - start) * 1000
            self.completed += 1
            self.consecutive_errors = 0
            self.ttfts.append(ttft if ttft is not None else total)
            self.totals.append(total)
            self.est_tokens.append(est_tokens)
            if server_tokens is not None:
                self.server_tokens.append(server_tokens)
                self.completed_tokens += server_tokens
                if abs(server_tokens - self.req.target_tokens) > self.req.target_tokens * TOKEN_TOLERANCE:
                    self.token_mismatch += 1
            else:
                self.completed_tokens += est_tokens
        except asyncio.CancelledError:
            raise
        except WorkloadError as exc:
            self.errors += 1
            self.last_error = str(exc)
            self.fail(f"{exc} (workload stopped: the request was rejected, so it would not repeat usefully)")
        except httpx.HTTPError as exc:
            self.errors += 1
            self.consecutive_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            if self.consecutive_errors >= 3:
                self.fail(f"vLLM connection failing repeatedly ({self.last_error})")
            else:
                await asyncio.sleep(0.5)
        finally:
            self.in_flight -= 1
            if not got_first:
                self.awaiting -= 1

    async def poll_scheduler(self, client):
        while True:
            sched = await fetch_scheduler(client, max_age=0)
            if sched.get("available"):
                self.samples += 1
                waiting, running = sched.get("waiting") or 0, sched.get("running") or 0
                self.max_waiting = max(self.max_waiting, waiting)
                self.max_running = max(self.max_running, running)
                self.max_kv = max(self.max_kv, sched.get("kv_usage") or 0.0)
                if waiting > 0:
                    self.waiting_samples += 1
            await asyncio.sleep(0.5)

    # -- reporting ---------------------------------------------------------
    def snapshot(self, scheduler):
        end = self.finished or time.monotonic()
        elapsed = end - self.started
        sv = self.server_tokens
        return {
            "state": self.state, "reason": self.reason, "message": self.message,
            "elapsed_s": round(elapsed, 1),
            "remaining_s": round(max(0.0, self.req.duration_s - elapsed), 1) if self.state == "running" else 0,
            "duration_s": self.req.duration_s, "concurrency": self.req.concurrency,
            "target_tokens": self.req.target_tokens, "max_output_tokens": self.req.max_output_tokens,
            "priority_header": "0",
            "in_flight": self.in_flight, "awaiting_first_token": self.awaiting,
            "submitted": self.submitted, "completed": self.completed,
            "errors": self.errors, "last_error": self.last_error,
            "verified_prompts": self.verified, "token_mismatches": self.token_mismatch,
            "prompt_tokens": {
                "tokenize_min": min(self.est_tokens) if self.est_tokens else None,
                "tokenize_max": max(self.est_tokens) if self.est_tokens else None,
                "server_min": min(sv) if sv else None, "server_max": max(sv) if sv else None,
                "server_avg": round(sum(sv) / len(sv)) if sv else None},
            "ttft_ms": {"p50": pct(self.ttfts, .5), "p95": pct(self.ttfts, .95), "max": pct(self.ttfts, 1)},
            "completion_ms_p50": pct(self.totals, .5),
            "prefill_tokens_per_s": round(self.completed_tokens / elapsed) if elapsed > 1 else None,
            "gap_ms": {"avg": round(sum(self.gaps) / len(self.gaps), 1) if self.gaps else None,
                       "max": round(max(self.gaps), 1) if self.gaps else None},
            "scheduler": dict(scheduler, max_waiting_seen=self.max_waiting, max_running_seen=self.max_running,
                              max_kv_usage_seen=round(self.max_kv, 4),
                              waiting_samples=self.waiting_samples, samples=self.samples),
        }


WORKLOAD = None
WORKLOAD_TASK = None


async def server_limits(client, model):
    limits = {"max_model_len": None, "kv_capacity_tokens": None}
    try:
        r = await client.get(f"{VLLM_URL}/v1/models", headers=upstream_headers(), timeout=5)
        for m in r.json().get("data", []):
            if m.get("id") == model:
                limits["max_model_len"] = m.get("max_model_len")
    except (httpx.HTTPError, ValueError):
        pass
    sched = await fetch_scheduler(client, max_age=0)
    limits["kv_capacity_tokens"] = sched.get("kv_capacity_tokens")
    return limits


@app.post("/api/workload/start")
async def workload_start(req: WorkloadRequest):
    global WORKLOAD, WORKLOAD_TASK
    if WORKLOAD and WORKLOAD.state in ("running", "stopping"):
        raise HTTPException(409, "A background workload is already running; stop it first.")
    async with httpx.AsyncClient() as client:
        try:
            model = await model_name(client)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            raise HTTPException(503, f"Cannot reach vLLM at {VLLM_URL}: {exc}")
        limits = await server_limits(client, model)
        # Cheap pre-flight: must be able to verify token counts before generating load.
        try:
            await tokenize_chat(client, model, "preflight")
        except WorkloadError as exc:
            raise HTTPException(503, str(exc))
    need = req.target_tokens + req.max_output_tokens + 64      # chat template overhead
    if limits["max_model_len"] and need > limits["max_model_len"]:
        raise HTTPException(422, f"{req.target_tokens}-token prompts exceed the server's max context "
                                 f"({limits['max_model_len']} tokens). Choose a smaller prompt size.")
    cap = limits["kv_capacity_tokens"]
    if cap and req.concurrency * need > cap * KV_SAFETY_FRACTION:
        raise HTTPException(422, f"{req.concurrency} x {need} tokens would need {req.concurrency * need:,} KV "
                                 f"tokens but the server's KV cache holds {cap:,} (safe limit "
                                 f"{int(cap * KV_SAFETY_FRACTION):,}). Lower concurrency or prompt size.")
    WORKLOAD = Workload(req, model, limits)
    WORKLOAD_TASK = asyncio.create_task(WORKLOAD.run())
    return {"ok": True, "limits": limits}


@app.post("/api/workload/stop")
async def workload_stop():
    if WORKLOAD:
        WORKLOAD.stop()
        if WORKLOAD_TASK:
            await asyncio.wait({WORKLOAD_TASK}, timeout=5)
    return {"ok": True}


@app.get("/api/workload/status")
async def workload_status():
    sched = await fetch_scheduler()
    if WORKLOAD is None:
        return {"state": "idle", "scheduler": sched, "max_seconds": MAX_WORKLOAD_SECONDS,
                "max_concurrency": MAX_CONCURRENCY}
    out = WORKLOAD.snapshot(sched)
    out["max_seconds"], out["max_concurrency"] = MAX_WORKLOAD_SECONDS, MAX_CONCURRENCY
    return out


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Run the vLLM Starvation Playground UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
