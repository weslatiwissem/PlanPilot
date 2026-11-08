"""M4 - skill matcher: which team member should take which story.

Step 1  similarity(story text, member skills)   -> matrix (stories x members)
        backend "embeddings": sentence-transformers cosine similarity (default)
                              -> this is the 'pre-fine-tune' baseline from the proposal.
                              Set SKILL_MODEL to a fine-tuned model path later (no code change).
        backend "keywords":   plain word overlap. Used in tests / if the model can't load.
Step 2  Hungarian algorithm (scipy) picks the best story->member pairing while
        penalising members who already carry a lot of points (load balancing).
Step 3  If the best similarity is below MATCH_MIN_CONFIDENCE the story is NOT assigned:
        it is flagged `needs_human` (proposal: low-confidence items go to a human).
"""
import logging
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from orchestrator.schemas import Member

log = logging.getLogger(__name__)

LOAD_WEIGHT = 0.5  # how strongly workload counts compared to skill fit
_model = None      # cached sentence-transformers model


def _profile(m: Member) -> str:
    return f"{m.name}: " + ", ".join(m.skills)


def _keyword_sims(texts: List[str], members: List[Member]) -> np.ndarray:
    """Score = share of a member's skills that appear in the story text (capped)."""
    sims = np.zeros((len(texts), len(members)))
    for i, text in enumerate(texts):
        low = text.lower()
        for j, m in enumerate(members):
            hits = sum(1 for s in m.skills if re.search(r"\b" + re.escape(s.lower()), low))
            sims[i, j] = min(1.0, hits / 2)
    return sims


def _embedding_sims(texts: List[str], members: List[Member]) -> np.ndarray:
    global _model
    from sentence_transformers import SentenceTransformer  # lazy: heavy import

    if _model is None:
        name = os.getenv("SKILL_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _model = SentenceTransformer(name)
    t = _model.encode(texts, normalize_embeddings=True)
    m = _model.encode([_profile(x) for x in members], normalize_embeddings=True)
    return np.clip(t @ m.T, 0.0, 1.0)  # cosine similarity (vectors are normalised)


def similarity_matrix(texts: List[str], members: List[Member]) -> Tuple[np.ndarray, str]:
    """Returns (matrix, model_version). Falls back to keywords if embeddings fail."""
    if os.getenv("SKILL_MATCHER", "embeddings") == "embeddings":
        try:
            version = "embeddings:" + os.getenv("SKILL_MODEL", "all-MiniLM-L6-v2")
            return _embedding_sims(texts, members), version
        except Exception as e:  # no internet / model missing / import error
            log.warning("embedding matcher unavailable (%s); using keyword fallback", e)
            return _keyword_sims(texts, members), "keywords-v1 (fallback)"
    return _keyword_sims(texts, members), "keywords-v1"


def assign(stories: List[Dict], members: List[Member]):
    """Returns ({story title: (member name or None, confidence)}, model_version)."""
    if not stories or not members:
        return {s["title"]: (None, 0.0) for s in stories}, "none"

    texts = [f'{s["epic"]}. {s["title"]}. {s["description"]}' for s in stories]
    sims, version = similarity_matrix(texts, members)

    # Hungarian needs one column per (member, slot): the k-th task given to a member
    # costs more than the (k-1)-th, which spreads work across the team.
    n, k = len(stories), len(members)
    avg_points = float(np.mean([s["points"] for s in stories]))
    cost = np.zeros((n, k * n))
    for j, m in enumerate(members):
        for slot in range(n):
            load = m.current_load + slot * avg_points
            penalty = LOAD_WEIGHT * load / m.capacity_points
            cost[:, j * n + slot] = -sims[:, j] + penalty
    rows, cols = linear_sum_assignment(cost)

    min_conf = float(os.getenv("MATCH_MIN_CONFIDENCE", "0.15"))
    result = {}
    for i, c in zip(rows, cols):
        j = c // n
        conf = float(sims[i, j])
        result[stories[i]["title"]] = (members[j].name if conf >= min_conf else None, conf)
    return result, version
