import atexit
from dataclasses import dataclass, field, fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


@dataclass
class StepResult:
    finished: list[tuple[int, list[int]]] = field(default_factory=list)   # (seq_id, completion token ids)
    new_tokens: list[tuple[int, int]] = field(default_factory=list)       # (seq_id, token) produced this step
    num_prefill_tokens: int = 0
    num_decode_seqs: int = 0
    batch_size: int = 0


class LLMEngine:

    def __init__(self, model, event_sink=None, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config, event_sink=event_sink)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def cancel_request(self, seq: Sequence) -> bool:
        # Only between steps: a sequence cannot be cancelled while its batch is executing.
        return self.scheduler.cancel(seq)

    def step(self) -> StepResult:
        seqs = self.scheduler.schedule()
        num_prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs if not seq.in_decode)
        num_decode_seqs = sum(1 for seq in seqs if seq.in_decode)
        before = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        token_ids = self.model_runner.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids)
        new_tokens = [(seq.seq_id, seq.last_token) for seq in seqs if seq.num_completion_tokens > before[seq.seq_id]]
        finished = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return StepResult(
            finished=finished,
            new_tokens=new_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_seqs=num_decode_seqs,
            batch_size=len(seqs),
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        seqs = [self.add_request(prompt, sp) for prompt, sp in zip(prompts, sampling_params)]
        outputs = {}
        for seq in seqs:
            if seq.status == SequenceStatus.REJECTED:
                outputs[seq.seq_id] = {"text": "", "token_ids": [], "status": "rejected"}
                pbar.update(1)
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            result = self.step()
            elapsed = perf_counter() - t
            if result.num_prefill_tokens > 0:
                prefill_throughput = result.num_prefill_tokens / elapsed
            if result.num_decode_seqs > 0:
                decode_throughput = result.num_decode_seqs / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in result.finished:
                outputs[seq_id] = {"token_ids": token_ids, "status": "finished"}
                pbar.update(1)
        pbar.close()
        ordered = []
        for seq in seqs:
            output = outputs.get(seq.seq_id, {"text": "", "token_ids": [], "status": seq.status.name.lower()})
            if output["status"] == "finished":
                output["text"] = self.tokenizer.decode(output["token_ids"])
            ordered.append(output)
        return ordered
