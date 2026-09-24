"""FastAPI application for the policy support service."""

from fastapi import FastAPI

from graph import assistant_graph
from schemas import AskRequest, AskResponse


app = FastAPI(title="Zepto Policy Support", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = assistant_graph.invoke({"query": request.query})
    return AskResponse.model_validate(result["response"])
