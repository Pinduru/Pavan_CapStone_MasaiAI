"""Request, response and graph-state definitions."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AssistantState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_documents: list[str]
    sources: list[str]
    response: dict
