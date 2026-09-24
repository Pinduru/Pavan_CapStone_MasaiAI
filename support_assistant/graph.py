"""LangGraph workflow for intent routing and policy retrieval."""

from __future__ import annotations

import json
import os

import requests
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from prompt import GENERAL_PROMPT, POLICY_PROMPT
from rag import get_policy_store
from schemas import AskResponse, AssistantState


POLICY_KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)


def mock_mode() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"


def call_provider(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required when MOCK_LLM=0.")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def parse_provider_response(prompt: str, fallback_sources: list[str]) -> AskResponse:
    current_prompt = prompt
    for attempt in range(3):
        raw_response = call_provider(current_prompt)
        try:
            parsed = json.loads(raw_response)
            validated = AskResponse.model_validate(parsed)
            if fallback_sources and not validated.sources:
                validated.sources = fallback_sources
            return validated
        except (json.JSONDecodeError, ValidationError) as error:
            if attempt == 2:
                break
            current_prompt = (
                prompt
                + "\nYour previous response did not match the required JSON schema. "
                + f"Validation error: {error}. Return only corrected JSON."
            )

    return AskResponse(
        answer="The response could not be validated after three attempts.",
        sources=fallback_sources,
        confidence=0.0,
    )


def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"]
    if mock_mode():
        query_lower = query.lower()
        intent = "policy_question" if any(word in query_lower for word in POLICY_KEYWORDS) else "general_question"
    else:
        classification_prompt = (
            "Classify this query as exactly policy_question or general_question. "
            "Policy questions concern Zepto delivery, returns, refunds, membership, tracking, "
            f"cancellation, gift cards or support hours. Query: {query}"
        )
        raw_intent = call_provider(classification_prompt).strip().lower()
        intent = "policy_question" if "policy_question" in raw_intent else "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    documents, sources = get_policy_store().search(state["query"], limit=3)
    if mock_mode():
        snippet = documents[0][:200]
        response = AskResponse(
            answer=f"Based on the retrieved context: {snippet}",
            sources=sources,
            confidence=1.0,
        )
    else:
        context = "\n\n".join(
            f"[{source}] {document}" for source, document in zip(sources, documents)
        )
        prompt = POLICY_PROMPT.format(context=context, query=state["query"])
        response = parse_provider_response(prompt, sources)
    return {
        "retrieved_documents": documents,
        "sources": sources,
        "response": response.model_dump(),
    }


def direct_answer(state: AssistantState) -> AssistantState:
    if mock_mode():
        response = AskResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
    else:
        prompt = GENERAL_PROMPT.format(query=state["query"])
        response = parse_provider_response(prompt, [])
    return {"sources": [], "response": response.model_dump()}


def route_by_intent(state: AssistantState) -> str:
    return state["intent"]


def build_graph():
    builder = StateGraph(AssistantState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer", direct_answer)
    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )
    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer", END)
    return builder.compile()


assistant_graph = build_graph()
