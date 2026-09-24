"""Tests for routing, retrieval and response validation."""

import os

from fastapi.testclient import TestClient

os.environ["MOCK_LLM"] = "1"

from main import app
from rag import get_policy_store


client = TestClient(app)


def test_all_documents_are_indexed() -> None:
    assert get_policy_store().collection.count() == 8


def test_policy_question_uses_retrieval() -> None:
    response = client.post("/ask", json={"query": "What is the delivery fee?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Based on the retrieved context:")
    assert body["sources"][0] == "doc_01"
    assert len(body["sources"]) == 3
    assert body["confidence"] == 1.0


def test_general_question_uses_direct_answer() -> None:
    response = client.post("/ask", json={"query": "What is the capital of Canada?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "I can only answer questions about Zepto policies right now.",
        "sources": [],
        "confidence": 1.0,
    }
