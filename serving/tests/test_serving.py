import pytest
from fastapi.testclient import TestClient

from serving.app import app
from serving.config import load_settings


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "two-tower"
    assert body["n_users"] == 943
    assert body["n_items"] == 1682
    assert body["uptime_s"] >= 0


def test_recommend_schema(client):
    r = client.post("/recommend", json={"user_id": 1, "k": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == 1
    assert body["k"] == 10
    items = body["items"]
    assert len(items) == 10
    # ranks are 1..10 in order, scores strictly non-increasing, ids unique
    assert [it["rank"] for it in items] == list(range(1, 11))
    scores = [it["score"] for it in items]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    assert len({it["item_id"] for it in items}) == 10
    assert all(isinstance(it["item_id"], int) for it in items)
    assert body["latency_ms"] > 0


def test_recommend_default_k(client):
    settings = load_settings()
    r = client.post("/recommend", json={"user_id": 42})
    assert r.status_code == 200
    assert r.json()["k"] == settings.top_k_default


def test_recommend_validation(client):
    # missing user_id
    assert client.post("/recommend", json={"k": 10}).status_code == 422
    # k out of range
    assert client.post("/recommend", json={"user_id": 1, "k": 0}).status_code == 422
    assert client.post("/recommend", json={"user_id": 1, "k": 1000}).status_code == 422
    # non-integer user_id
    assert client.post("/recommend", json={"user_id": "abc"}).status_code == 422


def test_recommend_unknown_user(client):
    r = client.post("/recommend", json={"user_id": 999999, "k": 10})
    assert r.status_code == 404
    assert "unknown user_id" in r.json()["detail"]


def test_recommend_determinism(client):
    payload = {"user_id": 196, "k": 20}
    first = client.post("/recommend", json=payload).json()
    second = client.post("/recommend", json=payload).json()
    assert first["items"] == second["items"]


def test_recommend_excludes_seen(client):
    # user 1's train-seen items must not appear in their recommendations.
    # Reuse the recommender the app already loaded (its in-memory index) --
    # loading a second copy in-process would double memory for no reason.
    from serving.app import recommender as rec

    uidx = int(rec.data["user_idx"][1])
    seen_idx = rec.data["seen"][uidx].nonzero().flatten().tolist()
    seen_ids = {int(rec.data["item_ids"][i]) for i in seen_idx}
    r = client.post("/recommend", json={"user_id": 1, "k": 50}).json()
    rec_ids = {it["item_id"] for it in r["items"]}
    assert rec_ids.isdisjoint(seen_ids)


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["recommend_requests"] >= 1
    assert body["recommend_mean_latency_ms"] is not None
    assert body["recommend_mean_latency_ms"] > 0
