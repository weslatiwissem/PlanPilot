"""Retrieval of similar past stories (Chroma).

Build the index once:   python -m agent1_scoping.index_stories
Eyeball the results:    python -m agent1_scoping.rag "login with Google"

If the index does not exist yet, retrieve_similar() returns [] and Agent 1 simply
plans without RAG (a warning is logged).
"""
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# IMPORTANT (Windows): import chromadb BEFORE pandas / torch. Loading pandas first makes
# onnxruntime (imported by chromadb) crash with an access violation on some machines.
import chromadb  # noqa: E402

# Chroma 0.5.5 + posthog>=6 logs "Failed to send telemetry event ... capture() takes 1
# positional argument" on every call, even with telemetry disabled. It is harmless noise.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

COLLECTION = "past_stories"
# Cosine distance cut-off: neighbours further away than this are ignored in prompts.
# Tune it by looking at `python -m agent1_scoping.rag "<query>"` output.
MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.6"))

_embedding_function = None


@dataclass
class SimilarStory:
    title: str
    points: int
    project: str
    distance: float


def _default_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        _embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        )
    return _embedding_function


def _client():
    if os.getenv("CHROMA_USE_SERVER") == "1":  # docker compose chroma service
        return chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
    return chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", ".chroma"))


def get_collection(embedding_function=None, create: bool = False):
    client = _client()
    names = [getattr(c, "name", c) for c in client.list_collections()]
    if not create and COLLECTION not in names:
        raise RuntimeError(
            "Story index not found. Build it with: python -m agent1_scoping.index_stories"
        )
    ef = embedding_function or _default_embedding_function()
    if create:
        return client.get_or_create_collection(
            name=COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"}
        )
    return client.get_collection(name=COLLECTION, embedding_function=ef)


def retrieve_similar_stories(query: str, k: int = 5, collection=None) -> List[SimilarStory]:
    col = collection or get_collection()
    res = col.query(query_texts=[query[:1000]], n_results=k)
    return [
        SimilarStory(
            title=meta["title"],
            points=int(meta["points"]),
            project=meta["project"],
            distance=float(dist),
        )
        for meta, dist in zip(res["metadatas"][0], res["distances"][0])
    ]


def retrieve_similar(description: str, k: int = 5) -> List[str]:
    """Used by agent.py: short strings to put in the plan-generation prompt."""
    try:
        stories = retrieve_similar_stories(description, k)
    except Exception as e:  # missing index, model download problem, ...
        log.warning("RAG disabled for this request: %s", e)
        return []
    return [f"{s.title} ({s.points} points)" for s in stories if s.distance <= MAX_DISTANCE]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "login with Google"
    for s in retrieve_similar_stories(query, k=8):
        print(f"{s.distance:.3f}  {s.points:>3} pts  [{s.project}]  {s.title}")
