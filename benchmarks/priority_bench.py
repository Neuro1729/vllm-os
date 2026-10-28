"""Online benchmark: low-priority agent sessions vs. high-priority interactive chat.

Agents are closed-loop and multi-turn: each turn's prompt is the previous context plus its
completion plus a simulated tool result, and the next turn is issued after a think time.
More agents than batch slots keep the batch saturated. Chat requests arrive open-loop
(Poisson) with long-tailed output lengths.

One configuration per process; results are appended to a JSONL file.
"""
import argparse
import json
import os
import signal
import sys
from random import Random
from time import perf_counter, sleep

import numpy as np

from nanovllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    p.add_argument("--label", required=True)
    p.add_argument("--out", default="results/priority_bench.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=60, help="seconds during which new work arrives")
    p.add_argument("--num-agents", type=int, default=12)
    p.add_argument("--agent-think-time", type=float, default=0.5)
    p.add_argument("--agent-max-context", type=int, default=2048)
    p.add_argument("--chat-rate", type=float, default=4.0, help="chat requests per second")
    p.add_argument("--uniform-priority", action="store_true",
                   help="give chat and agents the same priority (nobody tells the scheduler what matters)")
    # engine
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--num-kvcache-blocks", type=int, default=-1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--enforce-eager", action="store_true")
    # scheduling
    p.add_argument("--policy", default="fcfs", choices=["fcfs", "priority", "mlfq"])
    p.add_argument("--preemption-mode", default="recompute", choices=["recompute", "swap", "auto"])
    p.add_argument("--swap-space", type=float, default=0)
    p.add_argument("--swap-min-tokens", type=int, default=0)
    p.add_argument("--mlfq-level-tokens", default="256,512,1024")
    p.add_argument("--mlfq-priority-scale", type=float, default=1.0)
    p.add_argument("--mlfq-max-levels", default="", help="priority:level,... e.g. 0:1")
    p.add_argument("--preempt-after-steps", type=int, default=0)
    p.add_argument("--aging-steps", type=int, default=0)
    p.add_argument("--starvation-steps", type=int, default=0)
    p.add_argument("--min-resident-steps", type=int, default=0)
    return p.parse_args()


def ints(s):
    return tuple(int(x) for x in s.split(",") if x)


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else None


class Workload:

    def __init__(self, args):
        self.args = args
        self.rng = Random(args.seed)
        rng = self.rng
        self.chat = []    # (arrival offset, prompt, sampling params)
        t = 1.0
        while True:
            t += rng.expovariate(args.chat_rate)
            if t > args.duration:
                break
            prompt = [rng.randrange(10000) for _ in range(rng.randint(32, 128))]
            out = rng.randint(16, 64) if rng.random() < 0.7 else rng.randint(64, 128)
            self.chat.append((t, prompt, SamplingParams(temperature=0.6, max_tokens=out, ignore_eos=True,
                                                     priority=1 if args.uniform_priority else 0)))
        # agent i: next turn time and current context
        self.agents = [dict(next_t=rng.uniform(0, 1.0), context=self.new_session(), seq=None, turns=0)
                       for _ in range(args.num_agents)]

    def new_session(self):
        return [self.rng.randrange(10000) for _ in range(self.rng.randint(300, 600))]

    def agent_turn(self, agent):
        rng = self.rng
        if len(agent["context"]) > self.args.agent_max_context:
            agent["context"] = self.new_session()
        out = rng.randint(128, 384)
        return agent["context"], SamplingParams(temperature=0.6, max_tokens=out, ignore_eos=True, priority=1)

    def agent_done(self, agent, seq, now):
        tool_result = [self.rng.randrange(10000) for _ in range(self.rng.randint(50, 150))]
        agent["context"] = seq.token_ids + tool_result
        agent["next_t"] = now + self.args.agent_think_time
        agent["seq"] = None
        agent["turns"] += 1


def main():
    args = parse_args()
    # turn SIGTERM (e.g. from `timeout`) into a normal exit so atexit tears the engine down cleanly
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    llm = LLM(
        args.model, max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
        num_kvcache_blocks=args.num_kvcache_blocks, gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager, scheduling_policy=args.policy, preemption_mode=args.preemption_mode,
        swap_space=args.swap_space, swap_min_tokens=args.swap_min_tokens, mlfq_level_tokens=ints(args.mlfq_level_tokens),
        aging_steps=args.aging_steps,
        starvation_steps=args.starvation_steps, min_resident_steps=args.min_resident_steps,
        mlfq_priority_scale=args.mlfq_priority_scale, preempt_after_steps=args.preempt_after_steps,
        mlfq_max_levels={int(k): int(v) for k, v in (x.split(":") for x in args.mlfq_max_levels.split(",") if x)},
    )
    llm.generate([[1, 2, 3]], SamplingParams(max_tokens=8), use_tqdm=False)    # warm up
    sched = llm.scheduler
    for k in sched.stats:
        sched.stats[k] = 0

    wl = Workload(args)
    chat_seqs, agent_seqs = [], []
    chat_i = 0
    start = perf_counter()
    steps = 0
    while True:
        now = perf_counter() - start
        while chat_i < len(wl.chat) and wl.chat[chat_i][0] <= now:
            _, prompt, sp = wl.chat[chat_i]
            chat_seqs.append(llm.add_request(prompt, sp))
            chat_i += 1
        next_events = [wl.chat[chat_i][0]] if chat_i < len(wl.chat) else []
        for agent in wl.agents:
            if agent["seq"] is not None and agent["seq"].is_finished:
                wl.agent_done(agent, agent["seq"], now)
            if agent["seq"] is None and now < args.duration:
                if agent["next_t"] <= now:
                    agent["seq"] = llm.add_request(*wl.agent_turn(agent))
                    agent_seqs.append(agent["seq"])
                else:
                    next_events.append(agent["next_t"])
        if llm.is_finished():
            if not next_events or now >= args.duration and chat_i == len(wl.chat):
                break
            sleep(max(0.0, min(next_events) - now))
            continue
        llm.step()
        steps += 1
    makespan = perf_counter() - start

    def summarize(seqs):
        ttft = [s.first_token_time - s.arrival_time for s in seqs]
        e2e = [s.finish_time - s.arrival_time for s in seqs]
        tpot = [(s.finish_time - s.first_token_time) / (s.num_completion_tokens - 1)
                for s in seqs if s.num_completion_tokens > 1]
        return dict(n=len(seqs), ttft_p50=pct(ttft, 50), ttft_p90=pct(ttft, 90), ttft_p99=pct(ttft, 99),
                    ttft_max=max(ttft, default=None), e2e_p50=pct(e2e, 50), e2e_p99=pct(e2e, 99),
                    tpot_p50=pct(tpot, 50), tpot_p99=pct(tpot, 99),
                    output_tokens=sum(s.num_completion_tokens for s in seqs),
                    preempted=sum(s.num_preemptions > 0 for s in seqs))

    result = dict(
        label=args.label, args=vars(args), makespan=makespan, steps=steps,
        num_kvcache_blocks=llm.scheduler.block_manager.blocks.__len__(),
        num_cpu_blocks=llm.scheduler.block_manager.num_cpu_blocks,
        chat=summarize(chat_seqs), agent=summarize(agent_seqs), stats=dict(sched.stats),
        agent_turns=sum(a["turns"] for a in wl.agents),
    )
    result["output_tokens_per_s"] = (result["chat"]["output_tokens"] + result["agent"]["output_tokens"]) / makespan
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    c, a = result["chat"], result["agent"]
    print(f"[{args.label}] chat TTFT p50/p99 {c['ttft_p50']:.3f}/{c['ttft_p99']:.3f}s  e2e p99 {c['e2e_p99']:.2f}s | "
          f"agent turns {result['agent_turns']} e2e p50 {a['e2e_p50']:.2f}s | {result['output_tokens_per_s']:.0f} tok/s | "
          f"{result['stats']}")


if __name__ == "__main__":
    main()
