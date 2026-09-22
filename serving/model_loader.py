"""Load trained artifacts once at startup; serve from in-memory tensors.

The two-tower model is loaded from `twotower.pt`, item embeddings are
precomputed for the whole catalog once (encode_items over all items), and
every request then costs one user-tower forward pass plus one
user-vector × item-matrix matmul followed by an exact top-k. Deterministic:
fixed torch seed, single-threaded matmuls (torch_threads=1 in config),
and a stable item-index tie-break.
"""

from __future__ import annotations

import os
import sys

import torch

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models import ItemTower, TwoTower, UserTower  # noqa: E402
from serving.config import Settings, load_settings  # noqa: E402


class Recommender:
    """Thread-safe read-only recommender: model + precomputed item index."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        torch.manual_seed(self.settings.seed)
        # Thread knobs are process-global: setting them twice in one process
        # raises RuntimeError, so tolerate that (e.g. tests loading a second
        # recommender in-process after the app's lifespan already did).
        for _set in (lambda: torch.set_num_threads(self.settings.torch_threads),
                     lambda: torch.set_num_interop_threads(self.settings.torch_threads)):
            try:
                _set()
            except RuntimeError:
                pass

        model_path = self.settings.resolved_model_path
        dataset_path = os.path.join(self.settings.resolved_artifact_dir, "dataset.pt")
        for p, name in [(model_path, "twotower.pt"), (dataset_path, "dataset.pt")]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing required artifact {name} at {p}. "
                    "Artifacts are NOT retrained here; run `python train.py` first."
                )
        self.data = torch.load(dataset_path, weights_only=False)
        art = torch.load(model_path, weights_only=True)

        model = TwoTower(
            UserTower(len(self.data["user_ids"]), self.data["n_age"], self.data["n_occ"],
                      art["id_dim"], art["out_dim"]),
            ItemTower(len(self.data["item_ids"]), self.data["n_genres"], self.data["n_year"],
                      art["id_dim"], art["out_dim"]),
            temperature=art["temperature"],
        )
        model.load_state_dict(art["state_dict"])
        model.eval()
        self.model = model
        self.temperature = float(art["temperature"])
        self.n_users = len(self.data["user_ids"])
        self.n_items = len(self.data["item_ids"])

        with torch.no_grad():
            all_i = torch.arange(self.n_items)
            # (I, d): the precomputed in-memory item index, L2-normalized rows.
            self.item_vectors = model.encode_items(all_i, self.data["i_genres"],
                                                   self.data["i_year"]).contiguous()
            # tiny per-index tie-break so equal scores order deterministically
            self._tie_break = torch.arange(self.n_items, dtype=torch.float64) * 1e-9

        self.titles = {int(mid): t for mid, t in
                       zip(self.data["item_ids"], self.data["titles"])}

    def user_vector(self, user_id: int) -> torch.Tensor:
        """Encode a raw user id -> (d,) user vector. Raises KeyError for unknown ids."""
        try:
            uidx = int(self.data["user_idx"][user_id])
        except (KeyError, TypeError):
            raise KeyError(f"unknown user_id: {user_id}")
        with torch.no_grad():
            return self.model.encode_users(
                torch.tensor([uidx]),
                self.data["u_age"][[uidx]],
                self.data["u_gender"][[uidx]],
                self.data["u_occ"][[uidx]],
            ).squeeze(0)

    @torch.no_grad()
    def recommend(self, user_id: int, k: int = 10):
        """Return top-k (rank, item_id, score) for a raw user_id.

        Scores = dot(user_vec, item_vec) / temperature, matching evaluate.py.
        Items the user interacted with in training are excluded (score -inf).
        Tie-break: lower catalog index first -> fully deterministic.
        """
        uidx = int(self.data["user_idx"][user_id])  # KeyError -> 404 upstream
        u = self.model.encode_users(
            torch.tensor([uidx]),
            self.data["u_age"][[uidx]],
            self.data["u_gender"][[uidx]],
            self.data["u_occ"][[uidx]],
        ).squeeze(0)
        scores = (self.item_vectors @ u) / self.temperature      # (I,)
        scores = scores.to(torch.float64)
        scores[self.data["seen"][uidx]] = float("-inf")          # exclude train-seen
        scores = scores - self._tie_break                         # deterministic ties
        k = min(max(k, 1), self.n_items)
        top_vals, top_idx = torch.topk(scores, k, sorted=True)
        item_ids = self.data["item_ids"]
        return [
            (rank + 1, int(item_ids[i]), float(top_vals[rank]))
            for rank, i in enumerate(top_idx.tolist())
        ]


def load_recommender(settings: Settings | None = None) -> Recommender:
    return Recommender(settings or load_settings())
