"""Open-loop trace replay. Contains no GPU code; the engine is injected.

Works with both engine generations:
  * V1/V2: add_request returns the Sequence.
  * V0: add_request returns None; the Sequence is scheduler.waiting[-1].

Every per-request timestamp is seconds since the start of the replay. Token
timestamps are taken at the end of the step that produced the token.
"""
import time

IDLE_SLEEP_S = 0.005


def new_record(request):
    return {
        "request_id": request["request_id"],
        "group": request["group"],
        "arrival_s": request["arrival_s"],
        "arrival_time": None,
        "admit_time": None,
        "first_token_time": None,
        "finish_time": None,
        "n_tokens": None,
        "status": "unfinished",
        "preemptions": None,
        "token_times": [],
    }


def new_records(requests):
    return [new_record(r) for r in requests]


class Replay:
    """Replays `requests` against `engine` in real time.

    `make_params(max_tokens)` builds the sampling parameters for one request.
    `clock` and `sleep` are injectable so tests can run without waiting.
    """

    def __init__(self, engine, requests, make_params, timeout_s,
                 clock=time.perf_counter, sleep=time.sleep):
        self.engine = engine
        self.requests = requests
        self.make_params = make_params
        self.timeout_s = timeout_s
        self.clock = clock
        self.sleep = sleep
        self.records = new_records(requests)
        self.steps = 0
        self.max_running = 0
        self.max_concurrent_active = 0
        self.timed_out = False
        self.wall_time_s = 0.0
        self._t0 = None
        self._seqs = {}     # index -> Sequence, for every arrival that was accepted
        self._active = {}   # index -> Sequence, arrived and not yet terminal

    def _elapsed(self):
        return self.clock() - self._t0

    def run(self):
        self._t0 = self.clock()
        n = len(self.requests)
        next_idx = 0
        while True:
            now = self._elapsed()
            if now > self.timeout_s:
                self.timed_out = True
                break
            while next_idx < n and self.requests[next_idx]["arrival_s"] <= now:
                self._arrive(next_idx)
                next_idx += 1
            if self.engine.is_finished():
                if next_idx >= n:
                    break
                wait = self.requests[next_idx]["arrival_s"] - now
                self.sleep(min(IDLE_SLEEP_S, max(0.0, wait)))
                continue
            self.engine.step()
            self.steps += 1
            self._observe(self._elapsed())
        self.wall_time_s = self._elapsed()
        self._finalize()

    def _arrive(self, idx):
        request = self.requests[idx]
        record = self.records[idx]
        record["arrival_time"] = self._elapsed()
        ret = self.engine.add_request(request["prompt_token_ids"], self.make_params(request["max_tokens"]))
        seq = ret if ret is not None else self.engine.scheduler.waiting[-1]
        self._seqs[idx] = seq
        if seq.status.name == "REJECTED":
            record["status"] = "rejected"
            record["n_tokens"] = 0
            return
        self._active[idx] = seq
        self.max_concurrent_active = max(self.max_concurrent_active, len(self._active))

    def _observe(self, now):
        running_ids = {id(s) for s in self.engine.scheduler.running}
        self.max_running = max(self.max_running, len(running_ids))
        for idx, seq in list(self._active.items()):
            record = self.records[idx]
            finished = seq.status.name == "FINISHED"
            if record["admit_time"] is None and (id(seq) in running_ids or finished):
                record["admit_time"] = now
            produced = seq.num_completion_tokens - len(record["token_times"])
            if produced > 0:
                record["token_times"].extend([now] * produced)
                if record["first_token_time"] is None:
                    record["first_token_time"] = record["token_times"][0]
            if finished:
                record["status"] = "finished"
                record["finish_time"] = now
                record["n_tokens"] = seq.num_completion_tokens
                del self._active[idx]

    def _finalize(self):
        for idx, seq in self._seqs.items():
            record = self.records[idx]
            feedback = getattr(seq, "feedback", None)
            record["preemptions"] = feedback.preemptions if feedback is not None else None
            if record["status"] == "unfinished" and record["arrival_time"] is not None:
                record["n_tokens"] = seq.num_completion_tokens
