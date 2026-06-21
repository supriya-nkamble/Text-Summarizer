import pytest
from tests.conftest import SAMPLE_DIALOGUE, SAMPLE_SUMMARY


def test_root_redirects_to_docs(api_client):
    response = api_client.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
    assert response.headers["location"].endswith("/docs")


def test_predict_returns_summary(api_client):
    response = api_client.post("/predict", params={"text": SAMPLE_DIALOGUE})
    assert response.status_code == 200
    assert response.json() == SAMPLE_SUMMARY


def test_predict_rejects_short_input(api_client):
    response = api_client.post("/predict", params={"text": "Too short"})
    assert response.status_code == 422
    assert "too short" in response.json()["detail"].lower()


def test_predict_rejects_empty_input(api_client):
    response = api_client.post("/predict", params={"text": ""})
    assert response.status_code == 422


def test_predict_rejects_oversized_input(api_client):
    response = api_client.post("/predict", params={"text": "x" * 10_001})
    assert response.status_code == 422
    assert "too long" in response.json()["detail"].lower()


def test_predict_accepts_max_length_boundary(api_client):
    text = "word " * 2_000  # ~10 000 chars, exactly at limit
    response = api_client.post("/predict", params={"text": text})
    assert response.status_code == 200
