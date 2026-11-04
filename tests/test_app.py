"""HTTP contract tests with a local mock model server; no GPU required."""
import asyncio
import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("playground", ROOT / "app.py")
playground = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = playground
spec.loader.exec_module(playground)


@pytest.fixture()
def model_server(monkeypatch):
    mock = FastAPI()

    class Seen(list):
        opts = {}                     # per-test knobs: fail (HTTP status), delay, max_model_len
    seen = Seen()
    seen.opts = {"delay": 0.05}

    @mock.get("/v1/models")
    async def models():
        return {"data": [{"id": "demo-model", "max_model_len": seen.opts.get("max_model_len", 16384)}]}

    @mock.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            '# HELP x\nvllm:num_requests_running{engine="0",model_name="demo"} 3.0\n'
            'vllm:num_requests_waiting{engine="0",model_name="demo"} 5.0\n'
            'vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 5.0\n'
            'vllm:kv_cache_usage_perc{engine="0",model_name="demo"} 0.25\n'
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="1000",engine="0"} 1.0\n')

    @mock.post("/tokenize")
    async def tokenize(req: Request):
        data = await req.json()
        text = data.get("prompt") or "".join(m["content"] for m in data.get("messages", []))
        return {"count": len(text.split())}

    @mock.post("/v1/chat/completions")
    async def chat(req: Request):
        data = await req.json()
        seen.append({"priority": req.headers.get("X-Vllm-Priority"), "payload": data})
        if seen.opts.get("fail"):
            return PlainTextResponse("boom", status_code=seen.opts["fail"])

        async def events():
            await asyncio.sleep(seen.opts.get("delay", 0))
            for chunk in ("Hello", " from mock"):
                yield f'data: {json.dumps({"choices": [{"delta": {"content": chunk}}]})}\n\n'
                await asyncio.sleep(0)
            words = len(data["messages"][0]["content"].split())
            yield f'data: {json.dumps({"choices": [], "usage": {"prompt_tokens": words}})}\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(events(), media_type="text/event-stream")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(mock, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.03)
    assert server.started
    monkeypatch.setattr(playground, "VLLM_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(playground, "VLLM_MODEL", "")
    yield seen
    server.should_exit = True
    thread.join(timeout=5)


def test_health_and_html(model_server):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=playground.app), base_url="http://ui") as c:
            res = await c.get("/api/health")
            assert res.json()["model"] == "demo-model"
            assert res.json()["online"] is True
            res = await c.get("/")
            assert "vLLM Playground" in res.text
            assert "experimentTitle" in res.text
            assert (await c.get("/app.js")).status_code == 200
    asyncio.run(run())


def test_short_stream_priority(model_server):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=playground.app), base_url="http://ui", timeout=30) as c:
            res = await c.post("/api/chat", json={"prompt":"What is 2 + 2?", "kind":"short", "priority":2})
            events = [json.loads(line) for line in res.text.splitlines()]
            assert [e["event"] for e in events] == ["prepared", "started", "token", "token", "done"]
            assert events[-1]["ttft_ms"] is not None
            assert events[-1]["total_ms"] >= events[-1]["ttft_ms"]
            assert ''.join(e["text"] for e in events if e["event"]=="token") == "Hello from mock"
    asyncio.run(run())
    assert model_server[-1]["priority"] == "2"
    assert model_server[-1]["payload"]["stream"] is True


def test_long_stream_contains_real_text_and_is_unique(model_server):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=playground.app), base_url="http://ui", timeout=30) as c:
            payload = {"prompt":"Summarize the report.","kind":"long","priority":0,"target_tokens":2048}
            a,b = await asyncio.gather(
                c.post("/api/chat",json=dict(payload,seed="request-one")),
                c.post("/api/chat",json=dict(payload,seed="request-two")))
            assert a.status_code == b.status_code == 200
            a_events=[json.loads(x) for x in a.text.splitlines()]
            assert a_events[0]["event"]=="prepared"
            assert a_events[0]["prompt_chars"] > 10000
            assert a_events[0]["prompt_tokens"] > 1000
            assert a_events[-1]["event"] == "done"
    asyncio.run(run())
    assert len(model_server) == 2
    assert all(x["priority"] == "0" for x in model_server)
    a = model_server[0]["payload"]["messages"][0]["content"]
    b = model_server[1]["payload"]["messages"][0]["content"]
    assert a != b


# --------------------------- background P0 workload ---------------------------

@pytest.fixture(autouse=True)
def _reset_workload():
    playground.WORKLOAD = None
    playground.WORKLOAD_TASK = None
    playground._sched_cache.update(at=0.0, value={"available": False})
    yield


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=playground.app), base_url="http://ui", timeout=30)


def test_chat_reports_priority_header_and_server_token_count(model_server):
    async def run():
        async with client() as c:
            res = await c.post("/api/chat", json={"prompt": "What is 2 + 2?", "kind": "short", "priority": 2})
            events = [json.loads(x) for x in res.text.splitlines()]
            assert events[1]["event"] == "started" and events[1]["priority_header"] == "2"
            assert events[-1]["server_prompt_tokens"] is not None
    asyncio.run(run())
    assert model_server[-1]["priority"] == "2"
    assert model_server[-1]["payload"]["stream_options"] == {"include_usage": True}


def test_parse_metrics():
    m = playground.parse_metrics(
        'vllm:num_requests_running{engine="0"} 3.0\nvllm:num_requests_waiting{engine="0"} 5.0\n'
        'vllm:num_requests_waiting_by_reason{reason="x"} 9.0\nvllm:kv_cache_usage_perc{engine="0"} 0.25\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="1000"} 1.0\n')
    assert (m["running"], m["waiting"], m["kv_usage"], m["kv_capacity_tokens"]) == (3.0, 5.0, 0.25, 16000)


def test_workload_auto_replenishes_p0_unique_verified_prompts(model_server):
    async def run():
        async with client() as c:
            res = await c.post("/api/workload/start", json={"concurrency": 3, "target_tokens": 1024})
            assert res.status_code == 200, res.text
            assert (await c.post("/api/workload/start", json={"concurrency": 3})).status_code == 409
            await asyncio.sleep(1.5)
            st = (await c.get("/api/workload/status")).json()
            assert st["state"] == "running" and st["in_flight"] == 3 and st["concurrency"] == 3
            assert st["completed"] >= 6                      # continuous replacement, not one-shot
            assert st["scheduler"]["waiting"] == 5.0 and st["scheduler"]["available"]
            assert st["gap_ms"]["avg"] is not None and st["gap_ms"]["avg"] < 50
            assert st["token_mismatches"] == 0 and st["verified_prompts"] >= 6
            await c.post("/api/workload/stop")
            st = (await c.get("/api/workload/status")).json()
            assert st["state"] == "finished" and st["in_flight"] == 0 and st["reason"] == "stopped by user"
    asyncio.run(run())
    assert len(model_server) >= 6
    assert {x["priority"] for x in model_server} == {"0"}             # background load is P0 only
    prompts = [x["payload"]["messages"][0]["content"] for x in model_server]
    assert len(set(prompts)) == len(prompts)                          # every prompt unique
    for x in model_server:                                            # verified within 3% of target
        assert abs(len(x["payload"]["messages"][0]["content"].split()) - 1024) <= 1024 * 0.03


def test_workload_time_limit_stops_itself(model_server):
    async def run():
        async with client() as c:
            await c.post("/api/workload/start", json={"concurrency": 2, "target_tokens": 1024, "duration_s": 5})
            await asyncio.sleep(6.2)
            st = (await c.get("/api/workload/status")).json()
            assert st["state"] == "finished" and st["reason"] == "time limit reached" and st["in_flight"] == 0
    asyncio.run(run())


def test_workload_rejects_unsafe_requests(model_server):
    async def run():
        async with client() as c:
            model_server.opts["max_model_len"] = 2048
            res = await c.post("/api/workload/start", json={"concurrency": 2, "target_tokens": 4096})
            assert res.status_code == 422 and "max context" in res.json()["detail"]
            model_server.opts["max_model_len"] = 16384
            # mock KV capacity = 1000 blocks x 16 = 16,000 tokens -> 16 x 8K must be refused
            res = await c.post("/api/workload/start", json={"concurrency": 16, "target_tokens": 8192})
            assert res.status_code == 422 and "KV" in res.json()["detail"]
            assert (await c.post("/api/workload/start", json={"concurrency": 99})).status_code == 422
            assert (await c.get("/api/workload/status")).json()["state"] == "idle"
    asyncio.run(run())


def test_workload_surfaces_vllm_errors(model_server):
    async def run():
        async with client() as c:
            model_server.opts["fail"] = 500
            await c.post("/api/workload/start", json={"concurrency": 2, "target_tokens": 1024})
            await asyncio.sleep(1.0)
            st = (await c.get("/api/workload/status")).json()
            assert st["state"] == "error" and "HTTP 500" in st["message"] and st["in_flight"] == 0
    asyncio.run(run())


def test_long_prompt_endpoint_and_exact_send(model_server):
    async def run():
        async with client() as c:
            res = await c.post("/api/long-prompt", json={"prompt": "Summarize this.", "target_tokens": 2048})
            doc = res.json()
            assert res.status_code == 200
            assert abs(doc["prompt_tokens"] - 2048) <= 2048 * 0.05 and len(doc["prompt"]) > 5000
            assert doc["prompt"].startswith("Independent technical report")
            other = (await c.post("/api/long-prompt", json={"target_tokens": 2048})).json()
            assert other["prompt"] != doc["prompt"]                          # unique per request
            # What the user sees in the text box is what vLLM receives, byte for byte.
            sent = await c.post("/api/chat", json={"prompt": doc["prompt"], "exact": True,
                                                   "kind": "long", "priority": 0})
            events = [json.loads(x) for x in sent.text.splitlines()]
            assert events[0]["event"] == "prepared" and events[-1]["event"] == "done"
            return doc["prompt"]
    prompt = asyncio.run(run())
    assert model_server[-1]["payload"]["messages"][0]["content"] == prompt
    assert model_server[-1]["priority"] == "0"


def test_long_text_requires_exact_flag(model_server):
    async def run():
        async with client() as c:
            res = await c.post("/api/chat", json={"prompt": "x " * 2000, "kind": "long"})
            assert res.status_code == 422
    asyncio.run(run())
