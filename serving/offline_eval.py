"""Offline evaluation of the SERVED artifacts.

Loads twotower.pt / dataset.pt through the same model_loader the API uses and
recomputes Recall@10 / Recall@50 / NDCG@10 on the held-out last interaction per
user, using the exact protocol from evaluate.py (score ALL unseen items,
mask train-seen, 1-based rank of the held-out item).

Results are measured from the current artifacts; legacy numbers are not targets.

Usage:  python serving/offline_eval.py [--out serving/eval_results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SERVING_DIR, ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from serving.config import load_settings  # noqa: E402
from serving.model_loader import load_recommender  # noqa: E402




def metrics_for_ranks(ranks):
    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "Recall@10": float((ranks <= 10).mean()),
        "Recall@50": float((ranks <= 50).mean()),
        "NDCG@10": float(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SERVING_DIR, "eval_results.json"))
    args = ap.parse_args()
    t0 = time.time()

    rec = load_recommender(load_settings())
    data = rec.data
    test_u, test_i = data["test_u"], data["test_i"]
    n_test = len(test_u)
    print(f"[offline_eval] loaded two-tower ({rec.n_items} items), {n_test} test users")

    # Batched user encoding + full score matrix, same math as evaluate.py:
    # score = (u @ V^T) / temperature, train-seen masked to -inf.
    ranks = np.empty(n_test, dtype=np.int64)
    bs = 256
    with torch.no_grad():
        for s in range(0, n_test, bs):
            e = min(s + bs, n_test)
            idx = test_u[s:e]
            u = rec.model.encode_users(idx, data["u_age"][idx],
                                       data["u_gender"][idx], data["u_occ"][idx])
            scores = (u @ rec.item_vectors.t()) / rec.temperature
            seen = data["seen"][idx]
            scores = scores.masked_fill(seen, float("-inf"))
            order = scores.argsort(dim=1, descending=True, stable=True)
            pos = (order == test_i[s:e].unsqueeze(1)).nonzero()[:, 1]
            ranks[s:e] = (pos + 1).cpu().numpy()
            print(f"[offline_eval] scored {e}/{n_test} users", end="\r", flush=True)
    print()

    metrics = metrics_for_ranks(ranks)
    result = {
        "model": "two-tower",
        "n_test_users": n_test,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "implementation": "corrected genre parser and duplicate-positive mask",
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[offline_eval] wrote {args.out}")


if __name__ == "__main__":
    main()

