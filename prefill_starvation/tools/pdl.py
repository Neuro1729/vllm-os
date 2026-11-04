"""Parallel HTTP range downloader (network here is ~0.1-0.25 MB/s per connection).

usage: python3 pdl.py URL OUT [--conns 48]
"""
import argparse
import os
import sys
import threading
import time
import urllib.request

CHUNK = 4 * 1024 * 1024


def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.geturl(), int(r.headers["Content-Length"])


def fetch(url, out, start, end, tries=20):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError("short read")
            with open(out, "r+b") as f:
                f.seek(start)
                f.write(data)
            return
        except Exception as e:  # noqa: BLE001
            time.sleep(min(30, 2 ** i))
            last = e
    raise RuntimeError(f"range {start}-{end} failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--conns", type=int, default=48)
    a = ap.parse_args()
    url, size = head(a.url)
    tmp = a.out + ".part"
    done_marker = a.out + ".ranges"
    done = set(open(done_marker).read().split()) if os.path.exists(done_marker) else set()
    if not os.path.exists(tmp):
        with open(tmp, "wb") as f:
            f.truncate(size)
    ranges = [(s, min(size, s + CHUNK) - 1) for s in range(0, size, CHUNK)]
    todo = [r for r in ranges if f"{r[0]}" not in done]
    lock = threading.Lock()
    t0 = time.time()
    got = [0]

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                s, e = todo.pop()
            fetch(url, tmp, s, e)
            with lock:
                with open(done_marker, "a") as f:
                    f.write(f"{s}\n")
                got[0] += e - s + 1
                el = time.time() - t0
                sys.stdout.write(f"\r{a.out}: {got[0]/1e6:.0f}/{size/1e6:.0f} MB "
                                 f"{got[0]/1e6/max(el,1e-3):.1f} MB/s")
                sys.stdout.flush()

    ts = [threading.Thread(target=worker) for _ in range(a.conns)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    os.replace(tmp, a.out)
    os.remove(done_marker)
    print(f"\nDONE {a.out} {size} bytes in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
