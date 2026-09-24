"""Document ingestion, local embeddings and ChromaDB retrieval."""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


class PolicyStore:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._ingest_documents()

    def _ingest_documents(self) -> None:
        paths = sorted(DOCS_DIR.glob("doc_*.txt"))
        if len(paths) != 8:
            raise RuntimeError(f"Expected 8 policy documents, found {len(paths)}.")

        ids = [path.stem for path in paths]
        documents = [path.read_text(encoding="utf-8").strip() for path in paths]
        embeddings = self.model.encode(documents, normalize_embeddings=True).tolist()
        metadatas = [{"source": document_id} for document_id in ids]
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query: str, limit: int = 3) -> tuple[list[str], list[str]]:
        query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        sources = [metadata["source"] for metadata in result["metadatas"][0]]
        return documents, sources


_store: PolicyStore | None = None


def get_policy_store() -> PolicyStore:
    global _store
    if _store is None:
        _store = PolicyStore()
    return _store
