"""Latency/load smoke test for the serving API.

Starts `uvicorn serving.app:app` as a subprocess against the repo's artifacts,
fires 50 sequential requests (warm-up included) and then 20 concurrent
requests, and reports mean / p50 / p95 / p99 latency plus throughput.

Usage:  python serving/load_test.py [--sequential 50] [--concurrent 20]
                                     [--out serving/load_results.json]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import time

import httpx

SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SERVING_DIR, ".."))

USER_IDS = [1, 42, 100, 196, 303, 400, 512, 603, 700, 864]


def wait_for_health(base: str, timeout_s: float = 60.0):
    # trust_env=False: ignore the sandbox's egress-proxy env vars for localhost.
    # (httpx's no_proxy matching chokes on the bracketed IPv6 entries in
    # this environment's NO_PROXY list.)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(base + "/health", timeout=5, trust_env=False)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not become healthy in time")


def one_request(base: str, uid: int, k: int = 10) -> float:
    t0 = time.perf_counter()
    with httpx.Client(base_url=base, timeout=30, trust_env=False) as c:
        r = c.post("/recommend", json={"user_id": uid, "k": k})
        r.raise_for_status()
        assert len(r.json()["items"]) == k
    return (time.perf_counter() - t0) * 1000


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k)
    return xs[f] + (xs[f + 1] - xs[f]) * (k - f) if f + 1 < len(xs) else xs[f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequential", type=int, default=50)
    ap.add_argument("--concurrent", type=int, default=20)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--out", default=os.path.join(SERVING_DIR, "load_results.json"))
    args = ap.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_DIR + os.pathsep + env.get("PYTHONPATH", "")
    server_log = os.path.join(SERVING_DIR, "server.log")
    log_fh = open(server_log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.app:app",
         "--host", "127.0.0.1", "--port", str(args.port), "--workers", "1"],
        cwd=REPO_DIR, env=env,
        stdout=log_fh, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        wait_for_health(base)
        health = httpx.get(base + "/health", timeout=10, trust_env=False).json()
        print(f"[load_test] server up: {health['n_users']} users, "
              f"{health['n_items']} items")

        # warm-up (JIT-ish torch paths, model load behind us)
        for i in range(5):
            one_request(base, USER_IDS[i % len(USER_IDS)])

        seq_lat = [one_request(base, USER_IDS[i % len(USER_IDS)])
                   for i in range(args.sequential)]
        print(f"[load_test] {args.sequential} sequential done")

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as ex:
            futs = [ex.submit(one_request, base, USER_IDS[i % len(USER_IDS)])
                    for i in range(args.concurrent)]
            conc_lat = [f.result() for f in futs]
        wall_s = time.time() - t0
        print(f"[load_test] {args.concurrent} concurrent done")

        metrics = httpx.get(base + "/metrics", timeout=10, trust_env=False).json()
        all_lat = seq_lat + conc_lat
        result = {
            "config": {"sequential": args.sequential, "concurrent": args.concurrent,
                       "k": 10, "workers": 1, "host": "127.0.0.1",
                       "server": f"uvicorn@127.0.0.1:{args.port}"},
            "sequential_latency_ms": {
                "n": len(seq_lat), "mean": round(statistics.mean(seq_lat), 2),
                "p50": round(percentile(seq_lat, 50), 2),
                "p95": round(percentile(seq_lat, 95), 2),
                "p99": round(percentile(seq_lat, 99), 2),
                "min": round(min(seq_lat), 2), "max": round(max(seq_lat), 2),
            },
            "concurrent_latency_ms": {
                "n": len(conc_lat), "mean": round(statistics.mean(conc_lat), 2),
                "p50": round(percentile(conc_lat, 50), 2),
                "p95": round(percentile(conc_lat, 95), 2),
                "p99": round(percentile(conc_lat, 99), 2),
                "throughput_req_per_s": round(args.concurrent / wall_s, 1),
            },
            "server_metrics_after_run": metrics,
            "note": "localhost smoke test on CPU; latency dominated by the "
                    "user-tower forward pass + exact top-k over 1,682 items.",
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        print(f"[load_test] wrote {args.out}")
    finally:
        proc.terminate()
        proc.wait(timeout=15)
        log_fh.close()
        if proc.returncode not in (0, -15):
            print(f"[load_test] server exited with code {proc.returncode}; "
                  f"see {server_log}")


if __name__ == "__main__":
    main()
