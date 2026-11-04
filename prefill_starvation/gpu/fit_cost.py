"""Fit sim.CostModel-style step cost from a server VLLM_SCHED_TRACE: dt ~ base + tok + attn + n_dec."""
import json, sys
import numpy as np
S = [json.loads(l) for l in open(sys.argv[1]) if '"ev"' not in l]
X, Y = [], []
for a, b in zip(S, S[1:]):
    dt = (b['t'] - a['t']) * 1000
    if dt > 2000:
        continue  # idle gap between steps
    pf = a['pf']
    attn = sum(v[0] * (v[1] + v[0] / 2) for v in pf.values())
    X.append([1, a['tok'], attn, a['n_dec']]); Y.append(dt)
X = np.array(X, float); Y = np.array(Y)
coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
err = np.abs(X @ coef - Y) / np.maximum(Y, 1)
print('steps', len(Y), 'base_ms=%.2f per_tok_ms=%.5f attn_ms=%.3e per_dec_ms=%.4f' % tuple(coef))
print('median rel err %.1f%%  p90 %.1f%%' % (100 * np.median(err), 100 * np.percentile(err, 90)))
print('dt ms p50/p90/max', np.percentile(Y, 50).round(1), np.percentile(Y, 90).round(1), Y.max().round(1))
json.dump(dict(zip(['base_ms', 'per_tok_ms', 'attn_ms', 'per_dec_ms'], coef.tolist())), open(sys.argv[2], 'w'))
