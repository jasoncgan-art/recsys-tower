"""Download MovieLens 100k and write small processed CSVs.

Source: https://files.grouplens.org/datasets/movielens/ml-100k.zip
(943 users, 1682 movies, 100k ratings -- small enough to train on a laptop CPU.)

Design choice (documented): we keep only ratings >= 4 stars and treat them as
*positive implicit feedback* ("the user liked this enough to watchlist it").
Ratings 1-3 are dropped rather than used as negatives -- in implicit-feedback
retrieval, a low rating still means the user *saw* the item, which is a
different signal than "never interacted". Dropping them keeps the semantics
clean: every row in interactions.csv is a positive example.

Outputs (all under ./data/):
    data/ml-100k/          extracted raw files (u.data, u.item, u.user, ...)
    data/interactions.csv  user_id,item_id,rating,timestamp  (ratings >= 4 only)
    data/movies.csv        item_id,title,year,genres  (genres pipe-separated)
    data/users.csv         user_id,age,gender,occupation
"""

import csv
import io
import os
import urllib.request
import zipfile

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "ml-100k")

# u.item columns: id | title | release date | video release date | imdb url |
#                 unknown | Action | Adventure | Animation | Children's | Comedy |
#                 Crime | Documentary | Drama | Fantasy | Film-Noir | Horror |
#                 Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western
GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download_and_extract(url=DATA_URL, dest=RAW_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(os.path.join(dest, "u.data")):
        print(f"[download] raw data already present at {dest}, skipping download")
        return
    print(f"[download] fetching {url} ...")
    with urllib.request.urlopen(url) as resp:
        blob = resp.read()
    print(f"[download] got {len(blob) / 1e6:.1f} MB, extracting ...")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(DATA_DIR)
    print(f"[download] extracted to {dest}")


def write_interactions(raw=RAW_DIR, out=os.path.join(DATA_DIR, "interactions.csv")):
    """u.data is: user_id \\t item_id \\t rating \\t timestamp. Keep rating >= 4."""
    n_all, n_kept = 0, 0
    with open(os.path.join(raw, "u.data")) as f, open(out, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["user_id", "item_id", "rating", "timestamp"])
        for line in f:
            user_id, item_id, rating, timestamp = line.split()
            n_all += 1
            if int(rating) >= 4:  # positive implicit feedback
                w.writerow([user_id, item_id, rating, timestamp])
                n_kept += 1
    print(f"[process] interactions.csv: kept {n_kept}/{n_all} ratings >= 4")


def write_movies(raw=RAW_DIR, out=os.path.join(DATA_DIR, "movies.csv")):
    """Parse u.item (latin-1, '|' separated) into item_id,title,year,genres."""
    with open(os.path.join(raw, "u.item"), encoding="latin-1") as f, \
            open(out, "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["item_id", "title", "year", "genres"])
        for line in f:
            parts = line.rstrip("\n").split("|")
            item_id = parts[0]
            title = parts[1]
            year = parts[2].split("-")[-1] if parts[2] else ""
            genre_bits = parts[5:5 + len(GENRE_NAMES)]
            genres = "|".join(
                name for name, bit in zip(GENRE_NAMES, genre_bits) if bit == "1"
            )
            w.writerow([item_id, title, year, genres])
    print("[process] movies.csv written")


def write_users(raw=RAW_DIR, out=os.path.join(DATA_DIR, "users.csv")):
    """Parse u.user (user_id | age | gender | occupation | zip) -> drop zip."""
    with open(os.path.join(raw, "u.user")) as f, open(out, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["user_id", "age", "gender", "occupation"])
        for line in f:
            user_id, age, gender, occupation, _zip = line.rstrip("\n").split("|")
            w.writerow([user_id, age, gender, occupation])
    print("[process] users.csv written")


def main():
    download_and_extract()
    write_interactions()
    write_movies()
    write_users()
    print("[done] data ready in", DATA_DIR)


if __name__ == "__main__":
    main()
