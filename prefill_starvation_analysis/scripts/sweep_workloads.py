"""Workloads for the low-priority starvation sweep (shared by the CPU harness and the GPU client).

burst(N): 32 background decodes + N high-priority (prio 0) 16K-token prompts arriving together,
          + 20-token LOW-priority (prio 2) probes every 250 ms while the burst drains.
stream(rho): 32 decodes + Poisson stream of high-priority 16K prompts at `rho` x prefill
          capacity, + LOW-priority probes every 1 s.
`equal=True` gives every request priority 0 (the FCFS-equivalent control).
"""
import random

LONG, PROBE, N_DEC = 16000, 20, 32
PROBE_PRIO, HI_PRIO = 2, 0


def _decodes(prio):
    return [dict(id=i, arrival_ms=i * 20.0, prompt=200, output=6000, prio=prio, cls="bg",
                 src="decoder") for i in range(N_DEC)]


def burst(n_long, equal=False, cap_tok_s=9000.0, t_long=3000.0):
    pp = HI_PRIO if equal else PROBE_PRIO
    wl = _decodes(HI_PRIO)
    k = len(wl)
    for i in range(n_long):
        wl.append(dict(id=k, arrival_ms=t_long + i * 10.0, prompt=LONG, output=1, prio=HI_PRIO,
                       cls="long", src="long")); k += 1
    window_ms = max(6000.0, 1.4 * n_long * LONG / cap_tok_s * 1000.0)
    t = t_long + 100.0
    while t < t_long + 100.0 + window_ms:
        wl.append(dict(id=k, arrival_ms=t, prompt=PROBE, output=1, prio=pp, cls="short",
                       src="probe")); k += 1
        t += 250.0
    return wl


def stream(rho, equal=False, cap_tok_s=9000.0, dur_ms=60000.0, seed=0, t0=3000.0):
    pp = HI_PRIO if equal else PROBE_PRIO
    rnd = random.Random(seed)
    wl = _decodes(HI_PRIO)
    k = len(wl)
    lam = rho * cap_tok_s / LONG / 1000.0  # arrivals per ms
    t = t0
    while True:
        t += rnd.expovariate(lam)
        if t > t0 + dur_ms:
            break
        wl.append(dict(id=k, arrival_ms=t, prompt=LONG, output=1, prio=HI_PRIO, cls="long",
                       src="long")); k += 1
    t = t0 + 500.0
    while t < t0 + dur_ms:
        wl.append(dict(id=k, arrival_ms=t, prompt=PROBE, output=1, prio=pp, cls="short",
                       src="probe")); k += 1
        t += 1000.0
    return wl


# Scheduler settings compared (all with --scheduling-policy priority).
VARIANTS = {
    "nocap":    dict(thr=0,   adaptive=False, pcap=False, equal=False),
    "cap512":   dict(thr=512, adaptive=False, pcap=False, equal=False),
    "adaptive": dict(thr=512, adaptive=True,  pcap=False, equal=False),
    "cap256":   dict(thr=256, adaptive=False, pcap=False, equal=False),
    "equal":    dict(thr=512, adaptive=True,  pcap=False, equal=True),   # FCFS control
    "pr65283":  dict(thr=512, adaptive=True,  pcap=True,  equal=False),  # + priority budget cap
}
NS = [1, 2, 4, 8, 16]
RHOS = [0.5, 0.7, 0.85, 0.95]
