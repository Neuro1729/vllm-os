"""Drive the REAL vLLM V1 Scheduler (CPU only, no model) through a timed workload.

Every scheduling decision (who gets how many tokens each step, preemption, admission) is made by
vllm.v1.core.sched.scheduler.Scheduler. Only two things are faked:
  * the model runner: it returns 1 sampled token for each request whose scheduled tokens reach
    the end of its sequence (exactly what the GPU runner does, minus the math), and
  * the clock: each step's duration comes from cost_model.CostModel, which only decides *when*
    new requests arrive relative to steps (cpu_sweep.py uses a constant step time).
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,  # noqa: E402
                                        KVCacheGroupSpec)
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

from cost_model import CostModel  # noqa: E402

BS = 16
FAKE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_model")


def make_scheduler(budget, threshold, adaptive, policy, kv_tokens, max_num_seqs=1024, log_stats=False,
                   max_len=262144):
    model_config = ModelConfig(model=FAKE_MODEL, max_model_len=max_len,
                               skip_tokenizer_init=True, dtype="float16")
    sched_config = SchedulerConfig(
        max_num_batched_tokens=budget, max_num_seqs=max_num_seqs,
        long_prefill_token_threshold=threshold,
        long_prefill_token_threshold_adaptive=adaptive, policy=policy,
        enable_chunked_prefill=True, max_model_len=max_len, is_encoder_decoder=False)
    cache_config = CacheConfig(block_size=BS, enable_prefix_caching=False)
    vc = VllmConfig(model_config=model_config, scheduler_config=sched_config,
                    cache_config=cache_config)
    nblocks = kv_tokens // BS + 1
    vc.cache_config.num_gpu_blocks = nblocks
    kvc = KVCacheConfig(num_blocks=nblocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=BS, num_kv_heads=1,
                                                         head_size=64, dtype=torch.float16))])
    return Scheduler(vc, kvc, StructuredOutputManager(vc), block_size=BS, log_stats=log_stats)


def rid_str(i):
    return f"{i:09d}"  # vLLM breaks priority ties by request_id *string*; pad to keep int order


def run(workload, budget=8192, threshold=512, adaptive=True, policy="fcfs",
        kv_tokens=800_000, max_num_seqs=1024, cost: CostModel | None = None,
        t_end_ms=None, sample_every=50, verbose=True, tweak=None, max_steps=None):
    """workload: list of dicts {id, arrival_ms, prompt, output, prio, cls}.

    Returns (per_request: dict id->record, timeline: list of per-step samples, stats)."""
    cost = cost or CostModel()
    s = make_scheduler(budget, threshold, adaptive, policy, kv_tokens, max_num_seqs)
    for k, v in (tweak or {}).items():  # e.g. PR-65283: priority_prefill_budget_cap=True
        assert hasattr(s, k), k
        setattr(s, k, v)
    pending = sorted(workload, key=lambda w: (w["arrival_ms"], w["id"]))
    recs = {w["id"]: dict(w, t_first=None, t_finish=None, starved=0, max_starve_run=0,
                          cur_run=0, chunks=0, decode_skips=0, n_tokens_done=0)
            for w in workload}
    alive: set[int] = set()
    timeline = []
    t, step, pi = 0.0, 0, 0
    busy_ms = 0.0
    wall0 = time.time()
    while pi < len(pending) or alive:
        if t_end_ms is not None and t > t_end_ms:
            break
        if max_steps is not None and step >= max_steps:
            break
        while pi < len(pending) and pending[pi]["arrival_ms"] <= t:
            w = pending[pi]
            pi += 1
            s.add_request(Request(rid_str(w["id"]), [7] * w["prompt"],
                                  SamplingParams(max_tokens=max(1, w["output"]),
                                                 ignore_eos=True), None,
                                  arrival_time=w["arrival_ms"] / 1000.0, priority=w["prio"]))
            alive.add(w["id"])
        if not alive:
            t = pending[pi]["arrival_ms"]
            continue
        out = s.schedule()
        nst = out.num_scheduled_tokens
        if not nst:
            t = pending[pi]["arrival_ms"] if pi < len(pending) else t + 1.0
            continue
        step += 1
        chunks, dec_ctx, ntok, n_dec, n_pf = [], 0, 0, 0, 0
        sampled = []
        ids = list(nst.keys())
        for r in ids:
            req = s.requests[r]
            n = nst[r]
            after = req.num_computed_tokens
            before = after - n
            ntok += n
            if n == 1 and req.num_output_tokens > 0 and after >= req.num_tokens:
                dec_ctx += before
                n_dec += 1
            else:
                chunks.append((before, n))
                n_pf += 1
            sampled.append([1] if after >= req.num_tokens else [])
        dt = cost.step_ms(chunks, dec_ctx, ntok)
        t += dt
        busy_ms += dt
        # starvation bookkeeping (before outputs are applied)
        sched_ids = {int(r) for r in ids}
        n_waiting_prefill = 0
        for i in alive:
            rc = recs[i]
            if i in sched_ids:
                rc["cur_run"] = 0
                if rc["t_first"] is None:
                    rc["chunks"] += 1
                continue
            if rc["t_first"] is None:
                rc["starved"] += 1
                rc["cur_run"] += 1
                n_waiting_prefill += 1
                if rc["cur_run"] > rc["max_starve_run"]:
                    rc["max_starve_run"] = rc["cur_run"]
            else:
                rc["decode_skips"] += 1
        for r, smp in zip(ids, sampled):
            if smp:
                rc = recs[int(r)]
                if rc["t_first"] is None:
                    rc["t_first"] = t
        mro = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: k for k, r in enumerate(ids)},
                                sampled_token_ids=sampled)
        s.update_from_output(out, mro)
        for r, smp in zip(ids, sampled):
            if smp:
                i = int(r)
                if r not in s.requests:  # finished and freed
                    recs[i]["t_finish"] = t
                    alive.discard(i)
                    if recs[i].get("replenish"):  # closed loop: send a fresh copy now
                        nid = max(recs) + 1
                        w = {k: recs[i][k] for k in ("prompt", "output", "prio", "cls",
                                                     "src", "replenish")}
                        w.update(id=nid, arrival_ms=t)
                        recs[nid] = dict(w, t_first=None, t_finish=None, starved=0,
                                         max_starve_run=0, cur_run=0, chunks=0,
                                         decode_skips=0, n_tokens_done=0)
                        pending.insert(pi, w)
        if step % sample_every == 0:
            n_run, n_wait = s.get_request_counts()
            timeline.append({"t_ms": round(t, 1), "step": step, "running": n_run,
                             "waiting": n_wait, "decodes": n_dec, "prefill_chunks": n_pf,
                             "prefill_tokens": ntok - n_dec, "waiting_for_first_token":
                             n_waiting_prefill, "dt_ms": round(dt, 2)})
        if verbose and step % 5000 == 0:
            n_run, n_wait = s.get_request_counts()
            print(f"  step {step:7d} t={t/1000:8.1f}s running={n_run:4d} waiting={n_wait:5d} "
                  f"dec={n_dec:4d} pf_chunks={n_pf:3d} wall={time.time()-wall0:6.0f}s",
                  flush=True)
    stats = {"steps": step, "sim_time_s": t / 1000.0, "busy_s": busy_ms / 1000.0,
             "wall_s": time.time() - wall0, "unfinished": len(alive),
             "not_arrived": len(pending) - pi}
    return recs, timeline, stats
