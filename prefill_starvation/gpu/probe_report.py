"""Summarize probe (cls=short) starvation from analyze.py output. usage: probe_report.py A.json"""
import json, sys
d = json.load(open(sys.argv[1]))
P = [r for r in d['requests'] if r['cls'] == 'short' and r.get('ttft_s') is not None]
tl = d['timeline']
dts = d['step_dt_s']
srt = lambda k: sorted(r.get(k, 0) or 0 for r in P)
q = lambda a, p: a[min(len(a)-1, int(p*len(a)))]
st, zr, ws = srt('starved_steps'), srt('max_zero_run'), srt('wait_steps')
print(f"probes={len(P)}  starved_steps p50={q(st,.5)} max={st[-1]}  longest_zero_run max={zr[-1]}  wait_steps p50={q(ws,.5)} max={ws[-1]}")
win = [x for x in tl if x['t'] > 20]
if win:
    print(f"during test: steps={len(win)} median step={1000*sorted(dts)[len(dts)//2]:.0f}ms  avg decodes/step={sum(x['n_dec'] for x in win)/len(win):.0f}"
          f"  avg prefill chunks/step={sum(x['n_pf'] for x in win)/len(win):.1f}  max reqs waiting for 1st token={max(x['starved'] for x in win)}  cap={win[0]['cap']}"
          f"  preemptions={sum(x.get('preempted',0) for x in win)}  max kv_usage={max(x.get('kv_usage',0) for x in win):.2f}")
