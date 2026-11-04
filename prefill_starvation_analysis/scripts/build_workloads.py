"""Regenerate configs/workloads/*.json (the exact GPU workloads) from sweep_workloads.py.

The GPU workloads were sized assuming 9K prefill tokens/s; the RTX 3090 actually sustained
~13K tokens/s, so the *measured* load of the stream runs is lower than their file names
(see results/processed/summary_gpu.csv). usage: python scripts/build_workloads.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep_workloads import NS, burst, stream  # noqa: E402

out = os.path.join(HERE, "..", "configs", "workloads")
os.makedirs(out, exist_ok=True)
for n in NS:
    for eq in (False, True):
        json.dump(burst(n, equal=eq), open(f'{out}/burst{n}{"_eq" if eq else ""}.json', "w"))
for rho in (0.7, 0.85, 0.95):
    json.dump(stream(rho), open(f"{out}/stream{rho}.json", "w"))
print(sorted(os.listdir(out)))
