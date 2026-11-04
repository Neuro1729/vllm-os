"""Validate sim.py against the REAL vLLM 0.32 V1 Scheduler, step by step, on CPU.

No model / no GPU: we construct vllm.v1.core.sched.scheduler.Scheduler with a fake KV cache
config, feed it Requests, and fake the model runner (1 sampled token for every request whose
scheduled tokens reach the end of its sequence). Then we run the same scenario through sim.Sim
(with a 1-ms-per-step clock so arrivals line up by step) and compare num_scheduled_tokens per
request per step.

Run with the vllm-omni venv:  CUDA_VISIBLE_DEVICES= .../.venv/bin/python validate_vllm.py
"""
import glob
import json
import os
import sys

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import CostModel, Req, SchedConfig, Sim  # noqa: E402

BS = 16
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_model")  # config only


class UnitCost(CostModel):
    def step_ms(self, *a, **k):
        return 1.0


def make_real(budget, T, adaptive, policy, kv_tokens, max_len):
    model_config = ModelConfig(model=MODEL, max_model_len=max_len, skip_tokenizer_init=True,
                               dtype="float16")
    sched_config = SchedulerConfig(
        max_num_batched_tokens=budget, max_num_seqs=1024, long_prefill_token_threshold=T,
        long_prefill_token_threshold_adaptive=adaptive, policy=policy,
        enable_chunked_prefill=True, max_model_len=max_len, is_encoder_decoder=False)
    cache_config = CacheConfig(block_size=BS, enable_prefix_caching=False)
    vc = VllmConfig(model_config=model_config, scheduler_config=sched_config,
                    cache_config=cache_config)
    nblocks = kv_tokens // BS + 1  # +1: vLLM reserves the null block
    vc.cache_config.num_gpu_blocks = nblocks
    kvc = KVCacheConfig(num_blocks=nblocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=BS, num_kv_heads=1,
                                                         head_size=64, dtype=torch.float16))])
    return Scheduler(vc, kvc, StructuredOutputManager(vc), block_size=BS, log_stats=False)


def run_real(scn, budget, T, adaptive, policy, kv_tokens, n_steps, max_len):
    s = make_real(budget, T, adaptive, policy, kv_tokens, max_len)
    by_step = {}
    for (rid, step, plen, olen, prio) in scn:
        by_step.setdefault(step, []).append((rid, plen, olen, prio))
    log = []
    for k in range(n_steps):
        for rid, plen, olen, prio in by_step.get(k, []):
            s.add_request(Request(f"{rid:08d}", [7] * plen,
                                  SamplingParams(max_tokens=olen, ignore_eos=True), None,
                                  arrival_time=float(k), priority=prio))
        out = s.schedule()
        nst = {int(r): n for r, n in out.num_scheduled_tokens.items()}
        log.append(nst)
        ids = list(out.num_scheduled_tokens.keys())
        sampled = []
        for r in ids:
            req = s.requests[r]
            sampled.append([1] if req.num_computed_tokens >= req.num_tokens else [])
        mro = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                sampled_token_ids=sampled)
        s.update_from_output(out, mro)
    return log


def run_sim(scn, budget, T, adaptive, policy, kv_tokens, n_steps):
    cfg = SchedConfig(max_num_batched_tokens=budget, max_num_seqs=1024,
                      long_prefill_token_threshold=T, adaptive=adaptive, policy=policy,
                      kv_tokens=kv_tokens)
    # vLLM arrival at "step k" == added before schedule() call k (0-based). Sim adds requests
    # with arrival <= t before step; with 1ms steps, t before step k (0-based) is k.
    reqs = [Req(rid, float(step), plen, olen, prio) for (rid, step, plen, olen, prio) in scn]
    sim = Sim(cfg, UnitCost(), trace_rids={r.rid for r in reqs})
    log = []
    orig = sim._schedule_vllm

    def wrapped():
        sched = orig()
        log.append((int(round(sim.t)), {rid: n for rid, (r, n) in sched.items()}))
        return sched
    sim._schedule_vllm = wrapped
    sim.run(reqs, stop=lambda s, a, d: s.t >= n_steps)
    dense = [{} for _ in range(n_steps)]
    for t, alloc in log:  # sim skips idle steps; vLLM logs them as empty
        if t < n_steps:
            dense[t] = alloc
    return dense


def compare(name, scn, budget, T, adaptive, policy, kv_tokens, n_steps, max_len=131072):
    real = run_real(scn, budget, T, adaptive, policy, kv_tokens, n_steps, max_len)
    simv = run_sim(scn, budget, T, adaptive, policy, kv_tokens, n_steps)
    n = min(len(real), len(simv))
    mism = [(k, real[k], simv[k]) for k in range(n) if real[k] != simv[k]]
    total_tok = sum(sum(x.values()) for x in real[:n])
    print(f"[{'OK ' if not mism else 'MISMATCH'}] {name:55s} steps={n:5d} "
          f"tokens={total_tok:8d} mismatched_steps={len(mism)}", flush=True)
    for k, a, b in mism[:3]:
        diff = {r: (a.get(r), b.get(r)) for r in set(a) | set(b) if a.get(r) != b.get(r)}
        print("    step", k, "real vs sim:", dict(list(diff.items())[:8]))
    return {"name": name, "steps": n, "mismatched_steps": len(mism)}


def main():
    res = []
    for n_dec in (0, 300, 768):
        for budget, T, adaptive in ((1024, 512, False), (1024, 512, True), (1024, 0, False),
                                    (8192, 512, True)):
            if n_dec >= budget:
                continue
            for policy in ("fcfs", "priority"):
                scn = [(i, 0, 200, 10**6, 1) for i in range(n_dec)]
                scn += [(100001, 400, 10_000, 30, 1), (100002, 400, 20_000, 30, 1),
                        (100003, 405, 20, 10, 0)]
                res.append(compare(f"dec={n_dec} B={budget} T={T} adapt={adaptive} {policy}",
                                   scn, budget, T, adaptive, policy, 4_000_000, 700))
    # KV-pressure scenario: forces preemption paths
    import random
    rnd = random.Random(0)
    scn = [(i, rnd.randrange(0, 300), rnd.choice([50, 300, 2000, 9000]),
            rnd.randrange(5, 200), rnd.randrange(0, 3)) for i in range(250)]
    for policy in ("fcfs", "priority"):
        for adaptive in (False, True):
            res.append(compare(f"random250 KV=60K (preemptions) adapt={adaptive} {policy}",
                               scn, 2048, 512, adaptive, policy, 60_000, 1500))
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "results/validation.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
