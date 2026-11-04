"""Regenerate configs/workloads/*.json (the exact GPU workloads) from sweep_workloads.py.

Stream workloads are sized with the measured RTX 3090 prefill capacity (GPU_CAP_TOK_S); the
load reported in results is always the *measured* one. usage: python scripts/build_workloads.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep_workloads import GPU_CAP_TOK_S, GPU_RHOS, NS, burst, stream  # noqa: E402

out = os.path.join(HERE, "..", "configs", "workloads")
os.makedirs(out, exist_ok=True)
for n in NS:
    for eq in (False, True):
        json.dump(burst(n, equal=eq), open(f'{out}/burst{n}{"_eq" if eq else ""}.json', "w"))
for rho in GPU_RHOS:
    for eq in (False, True):
        json.dump(stream(rho, equal=eq, cap_tok_s=GPU_CAP_TOK_S),
                  open(f'{out}/stream{rho}{"_eq" if eq else ""}.json', "w"))
print(sorted(os.listdir(out)))
