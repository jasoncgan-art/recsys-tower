"""Train the three retrieval models.

Usage:
    python train.py                       # train all three models
    python train.py --models popularity   # only the popularity baseline
    python train.py --models mf,twotower  # only MF + two-tower

Artifacts (saved under ./artifacts/, gitignored):
    dataset.pt     shared split + features + index maps
    popularity.pt  item popularity counts (train positives)
    mf.pt          MFWithBiases state dict
    twotower.pt    TwoTower state dict

Determinism: seeds fixed for random/numpy/torch; DataLoader shuffling uses a
seeded torch.Generator. Re-running on the same machine gives the same numbers.
"""

import argparse
import os
import random
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    build_item_features,
    build_user_features,
    load_raw,
    sample_negatives,
    temporal_leave_last_out,
)
from models import MFWithBiases, ItemTower, TwoTower, UserTower

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")

SEED = 42
DEVICE = torch.device("cpu")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# shared data prep
# --------------------------------------------------------------------------- #
def prepare():
    """Build the split + features once; save everything downstream needs."""
    os.makedirs(ART_DIR, exist_ok=True)
    interactions, movies, users = load_raw()
    train_df, test_df = temporal_leave_last_out(interactions)
    ufeat = build_user_features(users)
    ifeat = build_item_features(movies)

    print(f"[data] train positives: {len(train_df)}, test users: {len(test_df)}, "
          f"users: {len(ufeat['user_ids'])}, items: {len(ifeat['item_ids'])}")

    def to_idx(df):
        return (
            torch.tensor([ufeat["user_idx"][u] for u in df["user_id"]], dtype=torch.long),
            torch.tensor([ifeat["item_idx"][i] for i in df["item_id"]], dtype=torch.long),
        )

    tr_u, tr_i = to_idx(train_df)
    te_u, te_i = to_idx(test_df)

    # seen mask: (n_users, n_items) bool -- train interactions to exclude at eval
    seen = torch.zeros(len(ufeat["user_ids"]), len(ifeat["item_ids"]), dtype=torch.bool)
    seen[tr_u, tr_i] = True

    bundle = {
        "user_ids": ufeat["user_ids"], "item_ids": ifeat["item_ids"],
        "user_idx": ufeat["user_idx"], "item_idx": ifeat["item_idx"],
        "titles": ifeat["titles"],
        "train_u": tr_u, "train_i": tr_i, "test_u": te_u, "test_i": te_i,
        "seen": seen,
        "u_age": torch.tensor(ufeat["age"]), "u_gender": torch.tensor(ufeat["gender"]),
        "u_occ": torch.tensor(ufeat["occupation"]),
        "n_age": ufeat["n_age_buckets"], "n_occ": ufeat["n_occupations"],
        "i_genres": torch.tensor(ifeat["genres"]), "i_year": torch.tensor(ifeat["year"]),
        "n_year": ifeat["n_year_buckets"], "n_genres": ifeat["genres"].shape[1],
        "train_df": train_df[["user_id", "item_id"]],
    }
    torch.save(bundle, os.path.join(ART_DIR, "dataset.pt"))
    return bundle


# --------------------------------------------------------------------------- #
# (a) popularity baseline
# --------------------------------------------------------------------------- #
def train_popularity(data):
    """No training -- rank unseen items by how many train positives they got."""
    set_seed()
    counts = torch.zeros(len(data["item_ids"]))
    counts.index_add_(0, data["train_i"], torch.ones(len(data["train_i"])))
    torch.save({"counts": counts}, os.path.join(ART_DIR, "popularity.pt"))
    top = counts.argsort(descending=True)[:5]
    print("[popularity] top items:",
          [data["titles"][i] for i in top.tolist()])


# --------------------------------------------------------------------------- #
# (b) matrix factorization with biases, mini-batch SGD + BCE
# --------------------------------------------------------------------------- #
def train_mf(data, dim=32, epochs=20, batch_size=2048, n_neg=4, lr=0.05):
    set_seed()  # re-seed: this model's init must not depend on what ran before
    labeled = sample_negatives(data["train_df"], data["item_ids"], n_neg=n_neg, seed=SEED)
    u = torch.tensor([data["user_idx"][x] for x in labeled["user_id"]], dtype=torch.long)
    i = torch.tensor([data["item_idx"][x] for x in labeled["item_id"]], dtype=torch.long)
    y = torch.tensor(labeled["label"].to_numpy(), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(u, i, y), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = MFWithBiases(len(data["user_ids"]), len(data["item_ids"]), dim=dim).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for bu, bi, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bu, bi), by)
            loss.backward()
            opt.step()
            total += loss.item() * len(by)
            n += len(by)
        print(f"[mf] epoch {ep:02d}/{epochs}  bce={total / n:.4f}")
    torch.save({"state_dict": model.state_dict(), "dim": dim},
               os.path.join(ART_DIR, "mf.pt"))


# --------------------------------------------------------------------------- #
# (c) two-tower neural retrieval with in-batch sampled softmax
# --------------------------------------------------------------------------- #
def build_pos_padded(train_u, train_i, n_users):
    """(n_users, max_pos) LongTensor of each user's train-positive item indices,
    padded with -1. Used to mask false negatives out of the in-batch softmax."""
    pos = defaultdict(list)
    for u, i in zip(train_u.tolist(), train_i.tolist()):
        pos[u].append(i)
    max_p = max(len(v) for v in pos.values())
    padded = torch.full((n_users, max_p), -1, dtype=torch.long)
    for u, lst in pos.items():
        padded[u, :len(lst)] = torch.tensor(lst, dtype=torch.long)
    return padded


def inbatch_neg_mask(bu, bi, pos_padded, n_items):
    """(B, B) bool mask: True where batch item j is a known train positive of
    batch user b (but not the pair's own positive on the diagonal)."""
    # Keep every batch occurrence of an item. A single item->column map
    # silently drops duplicates, leaving known positives in the denominator.
    if bu.numel() != bi.numel():
        raise ValueError("user and item batches must have equal lengths")
    known = torch.zeros((len(bu), n_items), dtype=torch.bool, device=bu.device)
    candidates = pos_padded[bu]
    rows = torch.arange(len(bu), device=bu.device).unsqueeze(1).expand_as(candidates)
    valid = candidates >= 0
    known[rows[valid], candidates[valid]] = True
    mask = known[:, bi]
    mask.fill_diagonal_(False)
    return mask


def train_twotower(data, id_dim=32, out_dim=64, epochs=25, batch_size=1024,
                   lr=5e-3, temperature=0.1):
    set_seed()  # re-seed: this model's init must not depend on what ran before
    n_users, n_items = len(data["user_ids"]), len(data["item_ids"])
    loader = DataLoader(
        TensorDataset(data["train_u"], data["train_i"]), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    pos_padded = build_pos_padded(data["train_u"], data["train_i"], n_users)
    # logQ correction (Yi et al., 2019): in-batch negatives are sampled
    # roughly proportional to item popularity Q(i); precompute log Q(i).
    counts = torch.zeros(n_items)
    counts.index_add_(0, data["train_i"], torch.ones(len(data["train_i"])))
    logQ = (counts / counts.sum()).clamp(min=1e-12).log()
    model = TwoTower(
        UserTower(n_users, data["n_age"], data["n_occ"], id_dim, out_dim),
        ItemTower(n_items, data["n_genres"], data["n_year"], id_dim, out_dim),
        temperature=temperature,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for bu, bi in loader:
            opt.zero_grad()
            u = model.user_tower(bu, data["u_age"][bu], data["u_gender"][bu], data["u_occ"][bu])
            v = model.item_tower(bi, data["i_genres"][bi], data["i_year"][bi])
            mask = inbatch_neg_mask(bu, bi, pos_padded, n_items)
            loss = model.inbatch_softmax_loss(u, v, neg_mask=mask,
                                              log_q=logQ[bi])
            loss.backward()
            opt.step()
            total += loss.item() * len(bu)
            n += len(bu)
        print(f"[twotower] epoch {ep:02d}/{epochs}  inbatch_softmax={total / n:.4f}")
    torch.save(
        {"state_dict": model.state_dict(), "id_dim": id_dim, "out_dim": out_dim,
         "temperature": temperature},
        os.path.join(ART_DIR, "twotower.pt"),
    )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="popularity,mf,twotower",
                    help="comma-separated subset of {popularity,mf,twotower}")
    args = ap.parse_args()
    which = {m.strip() for m in args.models.split(",")}

    set_seed()
    data = prepare()
    if "popularity" in which:
        train_popularity(data)
    if "mf" in which:
        train_mf(data)
    if "twotower" in which:
        train_twotower(data)
    print("[done] artifacts in", ART_DIR)


if __name__ == "__main__":
    main()

