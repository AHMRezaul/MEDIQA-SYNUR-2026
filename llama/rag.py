"""
Step 4: Concept-aware hybrid retrieval (RAG)

Goal:
Given a query transcript, retrieve top K training examples using a hybrid score:
  score = w_t * sim(transcript_emb, query_emb) + w_c * sim(concept_emb, query_emb_concept) + w_l * lexical_overlap

Where:
- transcript_emb: embedding of the training transcript
- concept_emb: embedding of concept_text for that training example (concept names joined)
- query_emb: embedding of the query transcript
- query_emb_concept: embedding of query-derived "concept query text" (optional; here we use transcript itself + extracted keywords)
- lexical_overlap: token overlap between query tokens and example concept tokens / transcript tokens (lightweight)

This step uses:
- KB built in Step 2 (KBExample list)
- SchemaIndex from Step 1/3 (for optional query concept keyword extraction, if desired)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from preprocess import KBExample  # from Step 2
from define import SchemaIndex  # from Step 1


# -----------------------------
# Text utilities
# -----------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")  # simple, fast
_WS_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


# -----------------------------
# Embedding backend
# -----------------------------

class SentenceTransformerEmbedder:
    """
    Thin wrapper around sentence-transformers.
    Uses L2-normalized embeddings so cosine similarity is dot product.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # local import
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)


# -----------------------------
# Query concept keyword hints (optional but helpful)
# -----------------------------

def extract_schema_keyword_hints(
    transcript: str,
    schema: SchemaIndex,
    max_hints: int = 30,
) -> List[str]:
    """
    Lightweight schema-aware keyword extraction:
    - Tokenize transcript
    - Add concept name tokens that appear in transcript (exact token overlap)
    - Also add enum tokens that appear (careful: can be noisy; we keep it conservative)

    Output: a list of hint strings (concept names / key tokens) to enrich the "concept query text".
    """
    tkns = set(tokenize(transcript))
    hints: List[str] = []

    # Only scan concept names (193 concepts: scanning is cheap)
    for cid in schema.all_ids():
        c = schema.get(cid)
        name_tokens = tokenize(c.name)
        if not name_tokens:
            continue
        # If at least one meaningful token matches, consider it a hint
        if any(nt in tkns for nt in name_tokens):
            hints.append(c.name)
            if len(hints) >= max_hints:
                break

    return hints


# -----------------------------
# Hybrid Retriever
# -----------------------------

@dataclass(frozen=True)
class RetrievalItem:
    example_id: str
    transcript: str
    gold_idvalues: Any  # keep as stored in KBExample
    gold_observations: Any
    score: float
    # diagnostics
    s_transcript: float
    s_concept: float
    s_lexical: float


class HybridConceptAwareRetriever:
    """
    Builds two embedding indices:
      - transcript embeddings for each KB example
      - concept_text embeddings for each KB example (concept names joined)
    Then scores examples using a weighted hybrid.

    Usage:
      retriever = HybridConceptAwareRetriever(kb, schema)
      retriever.build()
      top = retriever.retrieve(query_transcript, k=3)
    """

    def __init__(
        self,
        kb: List[KBExample],
        schema: Optional[SchemaIndex] = None,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        weights: Tuple[float, float, float] = (0.70, 0.25, 0.05),  # (w_t, w_c, w_l)
    ) -> None:
        self.kb = kb
        self.schema = schema
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.w_t, self.w_c, self.w_l = weights

        self._transcript_texts: List[str] = []
        self._concept_texts: List[str] = []
        self._transcript_emb: Optional[np.ndarray] = None
        self._concept_emb: Optional[np.ndarray] = None

        # Pre-tokenize KB for lexical overlap
        self._kb_transcript_tokens: List[List[str]] = []
        self._kb_concept_tokens: List[List[str]] = []

    def build(self, batch_size: int = 64) -> None:
        self._transcript_texts = [normalize_text(ex.transcript) for ex in self.kb]
        self._concept_texts = [normalize_text(ex.concept_text) for ex in self.kb]

        # Embeddings
        self._transcript_emb = self.embedder.encode(self._transcript_texts, batch_size=batch_size)
        self._concept_emb = self.embedder.encode(self._concept_texts, batch_size=batch_size)

        # Tokens for lexical overlap
        self._kb_transcript_tokens = [tokenize(t) for t in self._transcript_texts]
        self._kb_concept_tokens = [tokenize(ct) for ct in self._concept_texts]

    def _check_built(self) -> None:
        if self._transcript_emb is None or self._concept_emb is None:
            raise RuntimeError("Retriever not built. Call build() first.")

    def retrieve(
        self,
        query_transcript: str,
        k: int = 3,
        candidate_pool: int = 30,
        add_schema_hints_to_query: bool = True,
        max_schema_hints: int = 30,
    ) -> List[RetrievalItem]:
        """
        Two-stage retrieval for speed + stability:
        1) Candidate pool by transcript similarity (top candidate_pool)
        2) Re-rank those candidates using hybrid score (transcript + concept + lexical)

        Parameters:
        - k: final number of examples to return (2-3 recommended)
        - candidate_pool: how many transcript-nearest neighbors to consider for reranking
        - add_schema_hints_to_query: enrich concept query with schema-derived hints from transcript
        """
        self._check_built()

        q_text = normalize_text(query_transcript)
        q_emb = self.embedder.encode([q_text], batch_size=1)  # shape (1, d)
        q_emb = q_emb[0]

        # Candidate pool by transcript similarity
        # Because embeddings are normalized, cosine = dot product
        sims_t = np.dot(self._transcript_emb, q_emb)  # shape (N,)
        if candidate_pool >= len(sims_t):
            cand_idx = np.argsort(-sims_t)
        else:
            # Partial top-k selection
            cand_idx = np.argpartition(-sims_t, candidate_pool)[:candidate_pool]
            cand_idx = cand_idx[np.argsort(-sims_t[cand_idx])]

        # Build concept query text
        concept_query = q_text
        if add_schema_hints_to_query and self.schema is not None:
            hints = extract_schema_keyword_hints(q_text, self.schema, max_hints=max_schema_hints)
            if hints:
                concept_query = concept_query + " | " + " | ".join(hints)

        q_concept_emb = self.embedder.encode([concept_query], batch_size=1)[0]

        q_tokens = tokenize(q_text)

        items: List[RetrievalItem] = []
        for idx in cand_idx:
            # transcript similarity
            s_t = float(sims_t[idx])

            # concept similarity (concept_text embedding vs concept_query embedding)
            s_c = float(np.dot(self._concept_emb[idx], q_concept_emb))

            # lexical overlap: query tokens vs (concept tokens union transcript tokens)
            kb_tokens = self._kb_concept_tokens[idx]
            # A small lexical score that boosts cases where query shares concept words
            s_l = jaccard(q_tokens, kb_tokens)

            score = (self.w_t * s_t) + (self.w_c * s_c) + (self.w_l * s_l)

            ex = self.kb[idx]
            items.append(
                RetrievalItem(
                    example_id=ex.example_id,
                    transcript=ex.transcript,
                    gold_idvalues=ex.gold_idvalues,
                    gold_observations=ex.gold_observations,
                    score=score,
                    s_transcript=s_t,
                    s_concept=s_c,
                    s_lexical=s_l,
                )
            )

        # Sort by final hybrid score
        items.sort(key=lambda x: x.score, reverse=True)
        return items[:k]

