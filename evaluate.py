"""Evaluate all three models on the held-out last interaction per user.

Metrics (averaged over test users; each user has exactly ONE relevant item,
the movie they watched/liked last):
    Recall@K  = 1 if the held-out item is in the top-K else 0   (a hit rate)
    NDCG@10   = 1 / log2(rank + 1) if rank <= 10 else 0
                (ideal DCG is 1/log2(2) = 1 since there is a single relevant item)

For every user we score ALL items the user never interacted with in training
and rank them -- exactly what a retrieval stage does at serving time.
"""

import os
import time

import numpy as np
import torch

from models import MFWithBiases, ItemTower, TwoTower, UserTower

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")


def metrics_for_ranks(ranks):
    """ranks: array of 1-based ranks of the held-out item, one per user."""
    ranks = np.asarray(ranks, dtype=np.float64)
    recall10 = (ranks <= 10).mean()
    recall50 = (ranks <= 50).mean()
    ndcg10 = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0).mean()
    return {"Recall@10": recall10, "Recall@50": recall50, "NDCG@10": ndcg10}


def ranks_from_scores(scores, test_i, seen):
    """scores: (n_users, n_items). Mask train-seen items, then find each user's
    held-out item rank (1-based)."""
    scores = scores.clone()
    scores[seen] = float("-inf")
    order = scores.argsort(dim=1, descending=True)          # (U, I) item indices
    # position of the true item in each user's ranked list
    pos = (order == test_i.unsqueeze(1)).nonzero()[:, 1]   # 0-based
    return (pos + 1).cpu().numpy()


def eval_popularity(data):
    art = torch.load(os.path.join(ART_DIR, "popularity.pt"), weights_only=True)
    n_items = len(data["item_ids"])
    n_test = len(data["test_u"])
    # float64 + item-index tie-break: fully deterministic ordering.
    scores = (art["counts"].double().unsqueeze(0).expand(n_test, -1)
              - torch.arange(n_items, dtype=torch.float64).unsqueeze(0) * 1e-6)
    return ranks_from_scores(scores, data["test_i"], data["seen"][data["test_u"]])


def eval_mf(data):
    art = torch.load(os.path.join(ART_DIR, "mf.pt"), weights_only=True)
    model = MFWithBiases(len(data["user_ids"]), len(data["item_ids"]), dim=art["dim"])
    model.load_state_dict(art["state_dict"])
    model.eval()
    with torch.no_grad():
        p = model.p_user.weight            # (U, d)
        q = model.q_item.weight            # (I, d)
        scores = (model.mu
                  + model.b_user.weight.squeeze(-1).unsqueeze(1)
                  + model.b_item.weight.squeeze(-1).unsqueeze(0)
                  + p @ q.t())             # (U, I)
        scores = scores[data["test_u"]]    # keep test users only
    return ranks_from_scores(scores, data["test_i"], data["seen"][data["test_u"]])


def eval_twotower(data):
    art = torch.load(os.path.join(ART_DIR, "twotower.pt"), weights_only=True)
    model = TwoTower(
        UserTower(len(data["user_ids"]), data["n_age"], data["n_occ"],
                  art["id_dim"], art["out_dim"]),
        ItemTower(len(data["item_ids"]), data["n_genres"], data["n_year"],
                  art["id_dim"], art["out_dim"]),
        temperature=art["temperature"],
    )
    model.load_state_dict(art["state_dict"])
    model.eval()
    with torch.no_grad():
        all_i = torch.arange(len(data["item_ids"]))
        v_all = model.encode_items(all_i, data["i_genres"], data["i_year"])  # (I, d)
        u_all = model.encode_users(data["test_u"], data["u_age"][data["test_u"]],
                                   data["u_gender"][data["test_u"]],
                                   data["u_occ"][data["test_u"]])            # (T, d)
        scores = (u_all @ v_all.t()) / art["temperature"]
    return ranks_from_scores(scores, data["test_i"], data["seen"][data["test_u"]])


def main():
    t0 = time.time()
    data = torch.load(os.path.join(ART_DIR, "dataset.pt"), weights_only=False)

    results = {}
    for name, fn in [("popularity", eval_popularity),
                     ("mf", eval_mf),
                     ("two-tower", eval_twotower)]:
        ranks = fn(data)
        results[name] = metrics_for_ranks(ranks)
        print(f"[eval] {name:11s} done ({len(ranks)} test users)")

    print()
    print(f"{'model':<12} {'Recall@10':>10} {'Recall@50':>10} {'NDCG@10':>10}")
    print("-" * 46)
    for name in ["popularity", "mf", "two-tower"]:
        m = results[name]
        print(f"{name:<12} {m['Recall@10']:>10.4f} {m['Recall@50']:>10.4f} "
              f"{m['NDCG@10']:>10.4f}")
    print(f"\n[eval] finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
