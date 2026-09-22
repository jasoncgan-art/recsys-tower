"""FastAPI serving app for the two-tower retrieval model.

Endpoints:
    GET  /health     -> service + model status
    GET  /metrics    -> basic request counters
    POST /recommend  -> ranked (item_id, score) list for a user

Run:  uvicorn serving.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from serving.config import load_settings  # noqa: E402
from serving.model_loader import load_recommender  # noqa: E402

settings = load_settings()
recommender = None

_start_time = time.time()
_counters = {
    "health_requests": 0,
    "recommend_requests": 0,
    "recommend_errors": 0,
    "recommend_latency_sum_s": 0.0,
}
_lock = threading.Lock()


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="Raw MovieLens user id (1..943)")
    k: int | None = Field(
        default=None,
        ge=1,
        le=settings.top_k_max,
        description=f"Number of items to return (1..{settings.top_k_max})",
    )


class RecommendItem(BaseModel):
    rank: int
    item_id: int
    title: str | None
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    k: int
    items: list[RecommendItem]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    n_users: int
    n_items: int
    uptime_s: float


class MetricsResponse(BaseModel):
    uptime_s: float
    health_requests: int
    recommend_requests: int
    recommend_errors: int
    recommend_mean_latency_ms: float | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    recommender = load_recommender(settings)  # model + item index loaded once
    yield
    recommender = None


app = FastAPI(title="two-tower-recsys", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    with _lock:
        _counters["health_requests"] += 1
    return HealthResponse(
        status="ok",
        model="two-tower",
        n_users=recommender.n_users,
        n_items=recommender.n_items,
        uptime_s=time.time() - _start_time,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    with _lock:
        c = dict(_counters)
    mean_ms = (c["recommend_latency_sum_s"] / c["recommend_requests"] * 1000
               if c["recommend_requests"] else None)
    return MetricsResponse(
        uptime_s=time.time() - _start_time,
        health_requests=c["health_requests"],
        recommend_requests=c["recommend_requests"],
        recommend_errors=c["recommend_errors"],
        recommend_mean_latency_ms=mean_ms,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    k = req.k if req.k is not None else settings.top_k_default
    t0 = time.perf_counter()
    try:
        try:
            results = recommender.recommend(req.user_id, k)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"unknown user_id {req.user_id}: only trained users are "
                       "supported (cold-start users not implemented)",
            )
    except HTTPException:
        with _lock:
            _counters["recommend_errors"] += 1
        raise
    latency_ms = (time.perf_counter() - t0) * 1000
    with _lock:
        _counters["recommend_requests"] += 1
        _counters["recommend_latency_sum_s"] += latency_ms / 1000
    return RecommendResponse(
        user_id=req.user_id,
        k=k,
        items=[RecommendItem(rank=r, item_id=i,
                             title=recommender.titles.get(i), score=s)
               for r, i, s in results],
        latency_ms=latency_ms,
    )
