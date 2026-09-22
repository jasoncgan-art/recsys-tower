"""Top-N recommendations from the trained two-tower model.

Usage:
    python recommend.py                 # sample user, top 10
    python recommend.py --user_id 42 --top_n 15

Prints the user's most recent liked movies (from training data) alongside the
model's recommendations, so you can sanity-check that the picks are plausible.
"""

import argparse
import os

import torch

from dataset import load_raw
from models import ItemTower, TwoTower, UserTower

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")


def load_twotower():
    data = torch.load(os.path.join(ART_DIR, "dataset.pt"), weights_only=False)
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
    return model, data, art


@torch.no_grad()
def recommend(model, data, temperature, user_id, top_n=10):
    uidx = data["user_idx"][user_id]
    u = model.encode_users(
        torch.tensor([uidx]),
        data["u_age"][[uidx]], data["u_gender"][[uidx]], data["u_occ"][[uidx]],
    )  # (1, d)
    all_i = torch.arange(len(data["item_ids"]))
    v = model.encode_items(all_i, data["i_genres"], data["i_year"])  # (I, d)
    scores = (u @ v.t()).squeeze(0) / temperature
    scores[data["seen"][uidx]] = float("-inf")  # don't recommend seen items
    top = scores.argsort(descending=True)[:top_n]
    return [(int(i), float(scores[i])) for i in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", type=int, default=1)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    model, data, art = load_twotower()
    interactions, movies, _ = load_raw()
    title_of = dict(zip(movies["item_id"], movies["title"]))

    liked = (interactions[interactions["user_id"] == args.user_id]
             .sort_values("timestamp", ascending=False)["item_id"].head(5))
    print(f"\nUser {args.user_id} recently liked:")
    for mid in liked:
        print(f"  - {title_of.get(mid, mid)}")

    print(f"\nTop-{args.top_n} two-tower recommendations:")
    for rank, (iidx, score) in enumerate(
            recommend(model, data, art["temperature"], args.user_id, args.top_n), 1):
        mid = int(data["item_ids"][iidx])
        print(f"  {rank:2d}. {title_of.get(mid, mid)}  (score {score:.3f})")


if __name__ == "__main__":
    main()
