"""Clean GPU starvation tests: background decodes + long prompts + a stream of tiny probes.
usage: build_micro.py OUTDIR"""
import json, os, sys
out = sys.argv[1]; os.makedirs(out, exist_ok=True)

def build(n_dec, n_long, long_len=30000, probe_prio=0, long_prio=0, t_long=20000):
    wl, k = [], 0
    for i in range(n_dec):                      # background chat streams (keep decoding)
        wl.append(dict(id=k, arrival_ms=i * 10.0, prompt=200, output=4000, prio=long_prio,
                       cls='bg', src='decoder')); k += 1
    for i in range(n_long):                     # long-context prompts arrive together
        wl.append(dict(id=k, arrival_ms=t_long + i * 50.0, prompt=long_len, output=20,
                       prio=long_prio, cls='long', src='long')); k += 1
    for i in range(40):                         # 20-token probes, one every 0.5 s
        wl.append(dict(id=k, arrival_ms=t_long + 1000 + i * 500.0, prompt=20, output=5,
                       prio=probe_prio, cls='short', src='probe')); k += 1
    return wl

def lowprio_stream(n_hi=50, hi_len=16000, every_ms=1200.0, t0=5000.0):
    # strict priority, no aging: steady HIGH-priority long prompts, LOW-priority 20-token probes
    wl, k = [], 0
    for i in range(50):
        wl.append(dict(id=k, arrival_ms=i * 10.0, prompt=200, output=4000, prio=0, cls='bg',
                       src='decoder')); k += 1
    for i in range(n_hi):
        wl.append(dict(id=k, arrival_ms=t0 + i * every_ms, prompt=hi_len, output=20, prio=0,
                       cls='long', src='long')); k += 1
    for i in range(40):
        wl.append(dict(id=k, arrival_ms=t0 + 1000 + i * 1000.0, prompt=20, output=5, prio=1,
                       cls='short', src='probe')); k += 1
    return wl

cases = {
  'decodeheavy_base':  build(600, 0),
  'decodeheavy':       build(600, 4),
  'manyprefill_base':  build(50, 0),
  'manyprefill':       build(50, 8),
  'decodeheavy_inv':   build(600, 4, probe_prio=0, long_prio=1),   # probes HIGH priority
  'manyprefill_inv':   build(50, 8, probe_prio=0, long_prio=1),    # probes HIGH priority
  'decodeheavy_lowprobe': build(600, 4, probe_prio=1, long_prio=0),  # probes LOW, longs HIGH
  'manyprefill_lowprobe': build(50, 8, probe_prio=1, long_prio=0),   # probes LOW, longs HIGH
  'lowprio_stream':    lowprio_stream(),                           # probes LOW priority
}
for n, wl in cases.items():
    json.dump(wl, open(os.path.join(out, n + '.json'), 'w'))
    print(n, len(wl))
