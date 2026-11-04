"""Replay a workload against a running vLLM server and record per-request latency.

workload JSON: list of {id, arrival_ms, prompt, output, prio, cls, src}
Each request is sent at its arrival time (relative to start) to /v1/completions with
`prompt` = token ids of the given length, `max_tokens` = output, ignore_eos, stream.
Records send / first-token / finish wall times (time.time(), same clock as the server's
VLLM_SCHED_TRACE) to a JSONL file.

usage: replay_client.py WORKLOAD.json OUT.jsonl [--port 8011] [--priority] [--drain-s 600]
"""
import argparse
import asyncio
import json
import random
import time

import aiohttp

VOCAB_LO, VOCAB_HI = 1000, 150000  # Qwen2.5 vocab is 151936; stay clear of special tokens


async def one(session, url, w, pool, use_prio, t0, out_f, stats):
    delay = t0 + w["arrival_ms"] / 1000.0 - time.time()
    if delay > 0:
        await asyncio.sleep(delay)
    n = w["prompt"]
    ids = [VOCAB_LO + (w["id"] % 50000)] + pool[: n - 1]  # unique first token; no prefix cache
    body = {"model": "m", "prompt": ids, "max_tokens": int(w["output"]), "ignore_eos": True,
            "stream": True, "temperature": 0.0, "request_id": f"r{w['id']:09d}"}
    if use_prio:
        body["priority"] = int(w["prio"])
    rec = {"id": w["id"], "cls": w["cls"], "src": w.get("src", ""), "prompt": n,
           "output": w["output"], "prio": w["prio"], "arrival_ms": w["arrival_ms"],
           "t_send": time.time(), "t_first": None, "t_done": None, "n_chunks": 0, "err": None}
    stats["inflight"] += 1
    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                rec["err"] = f"http {resp.status}: {(await resp.text())[:200]}"
            else:
                async for raw in resp.content:
                    line = raw.decode(errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    if rec["t_first"] is None:
                        rec["t_first"] = time.time()
                    rec["n_chunks"] += 1
                rec["t_done"] = time.time()
    except asyncio.CancelledError:
        rec["err"] = "cancelled_at_cutoff"
    except Exception as e:  # noqa: BLE001
        rec["err"] = repr(e)[:200]
    finally:
        stats["inflight"] -= 1
        stats["done"] += 1
        stats.setdefault("cls_done", {}).setdefault(w["cls"], 0)
        stats["cls_done"][w["cls"]] += 1
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("out")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--priority", action="store_true", help="send each request's priority")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--stop-when-cls-done", default=None,
                    help="end the run (cancel the rest) once all requests of this cls finished")
    ap.add_argument("--drain-s", type=float, default=600.0,
                    help="after the last arrival, wait at most this long, then cancel")
    a = ap.parse_args()
    wl = json.load(open(a.workload))
    bad = [w["id"] for w in wl if w["prompt"] + w["output"] > a.max_model_len]
    if bad:  # never send positions beyond the model's context (device-side assert risk)
        raise SystemExit(f"{len(bad)} requests exceed max_model_len={a.max_model_len}: {bad[:5]}")
    rnd = random.Random(0)
    max_len = max(w["prompt"] for w in wl)
    pool = [rnd.randrange(VOCAB_LO, VOCAB_HI) for _ in range(max_len)]
    url = f"http://127.0.0.1:{a.port}/v1/completions"
    stats = {"inflight": 0, "done": 0}
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    conn = aiohttp.TCPConnector(limit=0)
    with open(a.out, "w") as out_f:
        async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
            t0 = time.time() + 2.0
            meta = {"meta": True, "t0": t0, "n": len(wl), "priority": a.priority}
            out_f.write(json.dumps(meta) + "\n")
            tasks = [asyncio.create_task(one(session, url, w, pool, a.priority, t0, out_f, stats))
                     for w in wl]
            last = t0 + max(w["arrival_ms"] for w in wl) / 1000.0
            n_target = sum(1 for w in wl if w["cls"] == a.stop_when_cls_done)
            while stats["done"] < len(wl):
                await asyncio.sleep(1)
                if a.stop_when_cls_done and \
                        stats.get("cls_done", {}).get(a.stop_when_cls_done, 0) >= n_target:
                    print("all target-class requests done -> cancelling the rest", flush=True)
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    break
                el = time.time() - t0
                print(f"t={el:7.1f}s done={stats['done']}/{len(wl)} inflight={stats['inflight']}",
                      flush=True)
                if time.time() > last + a.drain_s:
                    print("drain timeout -> cancelling remaining", flush=True)
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    break


if __name__ == "__main__":
    asyncio.run(main())
