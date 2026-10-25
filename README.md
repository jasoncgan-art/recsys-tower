# Two-Tower Movie Retrieval

A CPU-scale PyTorch retrieval project on MovieLens-100K: popularity, matrix
factorization and a two-tower model, plus a local FastAPI service.

## Run

```bash
pip install -r requirements.txt -r serving/requirements.txt
python download_data.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train.py
python evaluate.py
python serving/offline_eval.py
python -m pytest tests serving/tests -q
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

## What the system does

User ID + age bucket + gender + occupation -> user tower. Movie ID + genres + year
bucket -> item tower. Both produce normalized embeddings. Training uses in-batch
softmax, a sampling-frequency correction and known-positive masking. Inference
precomputes item vectors and performs exact scoring across the small catalog.

Ratings >=4 are positive feedback. Each user's last positive interaction is held
out; other positives train the models. The seen mask covers training positives,
NOT all historical low-rated interactions. One held-out relevant movie makes
Recall@K equivalent to hit rate in this particular evaluation.

Known-positive masking excludes every known-positive occurrence except each
row's designated positive, so duplicate items in a batch do not leak into the
softmax denominator. Genre features use MovieLens `u.item` columns starting at
6, where column 5 is the `unknown` flag and the final column is `Western`.
Evaluation and serving use stable sorting, so equal scores resolve by catalog
index without perturbing unequal scores.

## Results

Current measurements are in [`reports/evaluation.json`](reports/evaluation.json),
which also records the environment, the training-code hash and evaluation
limitations. Protocol: per-user leave-last-positive-out, seed 42, 942 test users.

| model | Recall@10 | Recall@50 | NDCG@10 |
|---|---|---|---|
| popularity | 0.0786 | 0.2304 | 0.0399 |
| MF | 0.0839 | 0.2325 | 0.0407 |
| two-tower | 0.1242 | 0.3355 | 0.0610 |

## Training modes

The default two-tower uses in-batch negatives only.

An opt-in hard-negative mode adds a second round, in `train.py`:

1. **Train normally** — in-batch softmax + `neg_mask` + `logQ`.
2. **Mine** — score every item with the trained model and take each user's
   highest-scoring items that are *not* that user's known train positives
   and *not* their held-out test item (`mine_hard_negatives` in `train.py`).
3. **Fine-tune on mixed negatives** — one softmax denominator containing
   both the in-batch negatives and the mined hard negatives, with
   `neg_mask` and `logQ` still active. Hard negatives are never used alone:
   the highest-scoring "negatives" are exactly where unlabeled positives
   hide, so a hard-only denominator could push real taste away.

```bash
python train.py --models twotower --hard-negatives   # hard-negative round, 5 fine-tune epochs
python train.py --models twotower --hard-negatives --hard-epochs 3 --n-hard 10
python evaluate.py                                   # same table, on the hard-negative artifact
```

The flag is off by default, so plain `python train.py` trains as described
above and reproduces the measurements in `reports/evaluation.json`. Running
with the flag overwrites `artifacts/twotower.pt` with the hard-negative model —
re-run plain `python train.py --models twotower` to restore the default artifact.

## Scope and limitations

- Evaluation is single-seed, with no validation-based hyperparameter search
  or uncertainty interval. Use an independent validation split before tuning.
- Per-user holdout does not prevent another user's later interactions from being
  in training; this is not a global chronological deployment simulation.
- No ANN index, second-stage ranker, online updates, monitoring or A/B experiment.
- Only ~1,682 catalog items; this does not establish million-item scalability.
- No cold-start fallback for unknown IDs.
- Serving load figures in `serving/load_results.json` are smoke-test figures,
  not sustained-load benchmarks.
