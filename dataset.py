"""Shared data loading: features, temporal leave-last-out split, negative sampling.

Everything in this module is deterministic (seeds set in train.py / evaluate.py;
numpy Generator objects are created with fixed seeds here).

Split protocol -- temporal leave-last-out (a.k.a. leave-one-out by time):
    * For each user, sort their positive interactions by timestamp.
    * Hold out the LAST interaction as the test target.
    * Everything earlier is training data.
  This mimics the real retrieval task: "given what the user liked in the past,
  retrieve what they will like next" within each user. Other users may have
  training events after this user's test timestamp; this is NOT a global
  chronological deployment simulation.

Users with fewer than 2 positive interactions are dropped (can't split them).
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
N_GENRES = len(GENRE_NAMES)

OCCUPATIONS = [
    "administrator", "artist", "doctor", "educator", "engineer",
    "entertainment", "executive", "healthcare", "homemaker", "lawyer",
    "librarian", "marketing", "none", "other", "programmer", "retired",
    "salesman", "scientist", "student", "technician", "writer",
]
AGE_BUCKETS = [17, 24, 34, 44, 49, 55, 200]  # right edges: 0-17, 18-24, ...


def age_bucket(age: int) -> int:
    for i, edge in enumerate(AGE_BUCKETS):
        if age <= edge:
            return i
    return len(AGE_BUCKETS) - 1


def year_bucket(year: int) -> int:
    """Bucket release year into ~decade-ish buckets (missing year -> 0)."""
    if year <= 0:
        return 0
    return min(1 + (year - 1900) // 10, 11)  # 0=unknown, 1..11 = 1900s..2000s+


def load_raw():
    """Load the processed CSVs written by download_data.py."""
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    return interactions, movies, users


def temporal_leave_last_out(interactions: pd.DataFrame, min_interactions: int = 2):
    """Split into train/test by holding out each user's last interaction by time.

    Returns (train_df, test_df). Both have columns [user_id, item_id, timestamp].
    """
    counts = interactions["user_id"].value_counts()
    keep = counts[counts >= min_interactions].index
    df = interactions[interactions["user_id"].isin(keep)].copy()
    df = df.sort_values(["user_id", "timestamp"])
    # The last row per user (by timestamp) becomes the test target.
    is_last = df.groupby("user_id").cumcount(ascending=False) == 0
    test_df = df[is_last].reset_index(drop=True)
    train_df = df[~is_last].reset_index(drop=True)
    return train_df, test_df


def build_user_features(users: pd.DataFrame):
    """Encode user side features for the user tower.

    Returns a dict with:
        user_ids   : array of raw user ids (index i -> user id)
        user_idx   : raw user id -> row index
        age        : int array of age-bucket ids
        gender     : int array (0=F, 1=M)
        occupation : int array of occupation ids
    """
    users = users.sort_values("user_id").reset_index(drop=True)
    occ_to_id = {o: i for i, o in enumerate(OCCUPATIONS)}
    gender_to_id = {"F": 0, "M": 1}
    feats = {
        "user_ids": users["user_id"].to_numpy(),
        "user_idx": {u: i for i, u in enumerate(users["user_id"].to_numpy())},
        "age": np.array([age_bucket(int(a)) for a in users["age"]], dtype=np.int64),
        "gender": np.array([gender_to_id[g] for g in users["gender"]], dtype=np.int64),
        "occupation": np.array(
            [occ_to_id.get(o, occ_to_id["other"]) for o in users["occupation"]],
            dtype=np.int64,
        ),
        "n_age_buckets": len(AGE_BUCKETS),
        "n_occupations": len(OCCUPATIONS),
    }
    return feats


def build_item_features(movies: pd.DataFrame):
    """Encode item side features for the item tower.

    Returns a dict with:
        item_ids : array of raw item ids (index i -> item id)
        item_idx : raw item id -> row index
        genres   : float32 multi-hot matrix (n_items, N_GENRES)
        year     : int array of year-bucket ids
    """
    movies = movies.sort_values("item_id").reset_index(drop=True)
    genre_to_id = {g: i for i, g in enumerate(GENRE_NAMES)}
    n = len(movies)
    genres = np.zeros((n, N_GENRES), dtype=np.float32)
    years = np.zeros(n, dtype=np.int64)
    for i, row in movies.iterrows():
        if isinstance(row["genres"], str) and row["genres"]:
            for g in row["genres"].split("|"):
                if g in genre_to_id:
                    genres[i, genre_to_id[g]] = 1.0
        try:
            years[i] = year_bucket(int(row["year"]))
        except (ValueError, TypeError):
            years[i] = 0
    return {
        "item_ids": movies["item_id"].to_numpy(),
        "item_idx": {m: i for i, m in enumerate(movies["item_id"].to_numpy())},
        "titles": movies["title"].to_numpy(),
        "genres": genres,
        "year": years,
        "n_year_buckets": 12,
    }


def user_seen_items(train_df: pd.DataFrame):
    """Map raw user_id -> set of raw item_ids the user interacted with in train."""
    seen = defaultdict(set)
    for u, i in zip(train_df["user_id"], train_df["item_id"]):
        seen[u].add(i)
    return seen


def sample_negatives(train_df, item_ids, n_neg: int, seed: int = 42):
    """For each train positive, sample `n_neg` negatives the user never saw.

    Negatives are drawn from ALL items minus the user's train interactions
    (this includes the held-out test item -- standard practice, and it only
    makes evaluation *harder*, never easier, since the model is pushed to
    score the test item low).

    Returns a DataFrame with columns [user_id, item_id, label].
    """
    rng = np.random.default_rng(seed)
    seen = user_seen_items(train_df)
    all_items = np.asarray(item_ids)
    rows = []
    for u, i in zip(train_df["user_id"].to_numpy(), train_df["item_id"].to_numpy()):
        rows.append((u, i, 1))
        unseen = np.array([m for m in all_items if m not in seen[u]])
        negs = rng.choice(unseen, size=n_neg, replace=len(unseen) < n_neg)
        for m in negs:
            rows.append((u, int(m), 0))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "label"])

