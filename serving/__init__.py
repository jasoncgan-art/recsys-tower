"""Production-style serving layer for the trained two-tower retrieval model.

Artifacts are loaded once at startup (item vectors precomputed), then every
request is a single user-tower forward pass + exact brute-force top-k over the
precomputed item index. See README's "Serving" section for the full picture.
"""
