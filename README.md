# Two-Tower Movie Retrieval: Correctness and Reproducibility

A CPU-scale PyTorch retrieval project on MovieLens-100K: popularity, matrix
factorization and a two-tower model, plus a local FastAPI service. This is a
personal learning project, not an industrial deployment or a full ranking system.

## Corrections in this revision

1. **Duplicate positive items in a batch:** the original one-item/one-column map
   skipped duplicate occurrences. The mask now excludes every known-positive
   occurrence except each row's designated positive. Regression tests cover this.
2. **Genre parser:** MovieLens `u.item` column 5 is `unknown`, not `Action`.
   Genre columns now start at 6 and include the final `Western` flag.
3. **Evaluation/serving ties:** stable sorting resolves equal scores by catalog
   index without perturbing unequal scores.
4. Old model results are superseded. Evaluation measures current checkpoints;
   it no longer treats hardcoded old numbers as required targets.

## Run

```bash
pip install -r requirements.txt -r serving/requirements.txt
python download_data.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train.py
python train.py --models twotower --hard-negatives  # opt-in v2 hard-negative round
python evaluate.py
python serving/offline_eval.py
python -m pytest tests serving/tests -q
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

Current measurements: [`reports/evaluation.json`](reports/evaluation.json).
The report includes environment, training-code hash and evaluation limitations.
Do not reuse the old 31% improvement claim without checking the new report.

## v2: hard negative mining (opt-in, off by default)

The default two-tower uses random in-batch negatives only. The v2
flag adds a second round, in `train.py`:

1. **Train normally** — in-batch softmax + `neg_mask` + `logQ`, exactly as above.
2. **Mine** — score every item with the trained model and take each user's
   highest-scoring items that are *not* that user's known train positives
   and *not* their held-out test item (`mine_hard_negatives` in `train.py`).
   Mining the test item would train on the answer, so it is excluded.
3. **Fine-tune on mixed negatives** — one softmax denominator containing
   both the random/in-batch negatives and the mined hard negatives, with
   `neg_mask` and `logQ` still active. Hard negatives are never used alone:
   the highest-scoring "negatives" are exactly where unlabeled positives
   hide, so a hard-only denominator could push real taste away.

```bash
python train.py --models twotower --hard-negatives   # v2 round, 5 fine-tune epochs
python train.py --models twotower --hard-negatives --hard-epochs 3 --n-hard 10
python evaluate.py                                   # same table, on the v2 artifact
```

**What does not change:** the flag is off by default, so plain
`python train.py` trains exactly as before and reproduces the current
measurements in `reports/evaluation.json`. Running with the flag overwrites
`artifacts/twotower.pt` with the v2 model — re-run plain
`python train.py --models twotower` to restore the default artifact.

**Status — read before citing:** this v2 code was added 2026-10-25 and has
*not* yet been studied or grill-passed by the repo owner. It is an opt-in
experiment, not a resume claim: do not list hard negative mining as
completed work until the owner has studied it under guidance and can defend
it line by line. No v2 metric is quoted here, because none has been measured
on a full training run yet — only a small smoke run proving the mining
path executes.

## What the system does

User ID + age bucket + gender + occupation -> user tower. Movie ID + genres + year
bucket -> item tower. Both produce normalized embeddings. Training uses in-batch
softmax, a sampling-frequency correction and known-positive masking. Inference
precomputes item vectors and performs exact scoring across the small catalog.

Ratings >=4 are positive feedback. Each user's last positive interaction is held
out; other positives train the models. The seen mask covers training positives,
NOT all historical low-rated interactions. One held-out relevant movie makes
Recall@K equivalent to hit rate in this particular evaluation.

## Scope and remaining work

- No validation-based hyperparameter search, multi-seed estimate or uncertainty
  interval is implemented yet. Use an independent validation split before tuning.
- Per-user holdout does not prevent another user's later interactions from being
  in training; this is not a global chronological deployment simulation.
- No ANN index, second-stage ranker, online updates, monitoring or A/B experiment.
- Only ~1,682 catalog items; this does not establish million-item scalability.
- No cold-start fallback for unknown IDs.

## Historical latency correction

`serving/load_results.json` is marked as a legacy smoke test, not a current model
benchmark. Its sequential p95 was 41.32 ms; its 20-request concurrent p95 was
661.11 ms with 29.2 req/s. These must never be combined into a claim of 41 ms at
20 concurrent requests. Sustained-load testing remains future work.

## Interview discussion

Explain how duplicate items become false negatives, how a one-column indexing
mistake changes features, and why correctness fixes require retraining. A bug fix
need not increase the selected test metric to be necessary. Distinguish the local
experiment from company production experience and distinguish retrieval from ranking.
