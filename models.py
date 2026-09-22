"""Model definitions.

1. MFWithBiases -- classic matrix factorization for implicit feedback:
       score(u, i) = mu + b_u + b_i + <p_u, q_i>
   Trained with binary cross-entropy over (positive, sampled negatives),
   optimized with mini-batch SGD (+momentum). This is the "shallow" neural
   baseline: no side features, just learned IDs + biases.

2. TwoTower -- neural retrieval model:
       user tower:  [user_id emb | age emb | gender emb | occupation emb]
                    -> MLP -> L2-normalized user vector
       item tower:  [item_id emb | genre multi-hot proj | year emb]
                    -> MLP -> L2-normalized item vector
       score(u, i) = <user_vec, item_vec> / temperature
   Trained with IN-BATCH SAMPLED SOFTMAX: within a batch of B (user, item)
   pairs, each user's item is the positive and the other B-1 items in the
   batch act as negatives. Loss = cross-entropy over the B x B score matrix.
   This is the workhorse loss behind YouTubeDNN-style retrieval and modern
   dual-encoder retrieval -- efficient (no explicit negative sampling) and a
   good approximation to full softmax when batches are large.

   What's handled, what's not (documented honestly):
     * Popularity bias of in-batch negatives: corrected with logQ
       (Yi et al., 2019) -- see the `log_q` argument below.
     * False negatives (a user's other liked items appearing as in-batch
       negatives): masked out of the softmax denominator -- see
       `inbatch_neg_mask` in train.py. Matters a lot on small catalogs.
     * Even so, in-batch negatives are still "easy" (random) negatives.
       Production systems additionally mine *hard* negatives from the
       current model; we skip that here for clarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFWithBiases(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 32):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1))          # global bias
        self.b_user = nn.Embedding(n_users, 1)          # user bias
        self.b_item = nn.Embedding(n_items, 1)          # item bias
        self.p_user = nn.Embedding(n_users, dim)        # user factors
        self.q_item = nn.Embedding(n_items, dim)        # item factors
        nn.init.normal_(self.p_user.weight, std=0.01)
        nn.init.normal_(self.q_item.weight, std=0.01)
        nn.init.zeros_(self.b_user.weight)
        nn.init.zeros_(self.b_item.weight)

    def forward(self, user_idx, item_idx):
        return (
            self.mu
            + self.b_user(user_idx).squeeze(-1)
            + self.b_item(item_idx).squeeze(-1)
            + (self.p_user(user_idx) * self.q_item(item_idx)).sum(-1)
        )


class UserTower(nn.Module):
    def __init__(self, n_users, n_age, n_occ, id_dim=32, out_dim=64):
        super().__init__()
        self.emb_user = nn.Embedding(n_users, id_dim)
        self.emb_age = nn.Embedding(n_age, 8)
        self.emb_gender = nn.Embedding(2, 4)
        self.emb_occ = nn.Embedding(n_occ, 8)
        self.mlp = nn.Sequential(
            nn.Linear(id_dim + 8 + 4 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, user_idx, age, gender, occ):
        x = torch.cat(
            [self.emb_user(user_idx), self.emb_age(age),
             self.emb_gender(gender), self.emb_occ(occ)],
            dim=-1,
        )
        return F.normalize(self.mlp(x), p=2, dim=-1)  # unit vectors -> cosine sim


class ItemTower(nn.Module):
    def __init__(self, n_items, n_genres, n_year, id_dim=32, out_dim=64):
        super().__init__()
        self.emb_item = nn.Embedding(n_items, id_dim)
        self.genre_proj = nn.Linear(n_genres, 16)  # multi-hot -> dense
        self.emb_year = nn.Embedding(n_year, 8)
        self.mlp = nn.Sequential(
            nn.Linear(id_dim + 16 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, item_idx, genres, year):
        x = torch.cat(
            [self.emb_item(item_idx), self.genre_proj(genres),
             self.emb_year(year)],
            dim=-1,
        )
        return F.normalize(self.mlp(x), p=2, dim=-1)


class TwoTower(nn.Module):
    def __init__(self, user_tower: UserTower, item_tower: ItemTower,
                 temperature: float = 0.1):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature

    def forward(self, u_idx, age, gender, occ, i_idx, genres, year):
        u = self.user_tower(u_idx, age, gender, occ)   # (B, d)
        v = self.item_tower(i_idx, genres, year)      # (B, d)
        return u, v

    def inbatch_softmax_loss(self, u, v, neg_mask=None, log_q=None):
        """u, v: (B, d) L2-normalized. Positive for row b is column b.

        neg_mask (optional): (B, B) bool -- True where column j is a *known*
        positive of user b (a false negative). Those logits are set to -inf
        so the model isn't punished for ranking a user's other liked items
        highly. (With a small catalog like MovieLens 100k, unmasked
        in-batch negatives are ~3-4% false negatives per row -- enough to
        noticeably hurt learning.)

        log_q (optional): (B,) tensor with log Q(item_j) for each candidate
        column j. In-batch negatives are sampled roughly proportional to
        item popularity, so without correction the model learns
        popularity-discounted scores. Subtracting log Q (Yi et al., 2019)
        removes that bias and the learned dot products directly approximate
        log P(item | user) -- which is what we rank by at serving time.
        """
        logits = (u @ v.t()) / self.temperature       # (B, B)
        if log_q is not None:
            logits = logits - log_q.unsqueeze(0)      # sampled-softmax correction
        if neg_mask is not None:
            logits = logits.masked_fill(neg_mask, float("-inf"))
        labels = torch.arange(u.size(0), device=u.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def encode_users(self, u_idx, age, gender, occ):
        return self.user_tower(u_idx, age, gender, occ)

    @torch.no_grad()
    def encode_items(self, i_idx, genres, year):
        return self.item_tower(i_idx, genres, year)
