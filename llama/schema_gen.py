"""
STEP 5: Candidate concept generation (schema-driven pruning)

Implements the recommended "must-include" concept IDs (high-FN concepts)
so they always appear in the candidate table, improving recall.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from define import SchemaIndex, ConceptDef
from rag import RetrievalItem, SentenceTransformerEmbedder


# -----------------------------
# Text utilities
# -----------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_WS_RE = re.compile(r"\s+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "for", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "as", "from", "that", "this",
    "it", "its", "patient", "pt", "clinician", "uh", "um"
}

def normalize_text(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]

def meaningful_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def token_set(s: str) -> Set[str]:
    return set(meaningful_tokens(tokenize(s)))


# -----------------------------
# Candidate concept types
# -----------------------------

@dataclass(frozen=True)
class CandidateConcept:
    id: str
    name: str
    value_type: str
    value_enum: Optional[List[str]]

@dataclass(frozen=True)
class CandidateReason:
    concept_id: str
    reason: str
    score: float

@dataclass(frozen=True)
class CandidateResult:
    candidate_ids: Set[str]
    candidates: List[CandidateConcept]
    reasons: Dict[str, List[CandidateReason]]  # id -> list of reasons


# -----------------------------
# Common concept set (optional)
# -----------------------------

def default_common_concept_ids(schema: SchemaIndex) -> Set[str]:
    patterns = [
        "heart rate", "pulse", "respiratory rate", "oxygen saturation", "temperature",
        "blood pressure", "pain", "weight", "height", "mental status", "work of breathing",
        "dyspnea", "cough", "nausea", "vomiting", "appetite", "intake", "urine", "bowel",
    ]
    out: Set[str] = set()
    for cid in schema.all_ids():
        name = schema.get(cid).name.lower()
        if any(p in name for p in patterns):
            out.add(cid)
    return out


# -----------------------------
# NEW: Must-include high-impact IDs
# -----------------------------

def default_force_include_ids(schema: SchemaIndex) -> Set[str]:
    """
    High-FN concepts from your dev error analysis (IDs known in your schema).
    These will be included even if lexical/semantic signals are weak.
    """
    must_ids = {"31", "40", "148", "130", "89", "117", "21", "67"}

    out: Set[str] = set()
    for cid in must_ids:
        if schema.has_id(cid):
            out.add(cid)
    return out


# -----------------------------
# Lexical scoring against schema
# -----------------------------

def lexical_score_concept(transcript_tokens: Set[str], concept: ConceptDef) -> Tuple[float, int, int]:
    name_toks = token_set(concept.name)
    name_ov = len(transcript_tokens & name_toks)

    score = 0.0
    if name_ov > 0:
        score += 1.0 + math.log(1 + name_ov)

    enum_ov = 0
    if concept.value_enum and concept.value_type in ("SINGLE_SELECT", "MULTI_SELECT"):
        enum_tokens = set()
        for v in concept.value_enum[:200]:
            enum_tokens |= token_set(str(v))
        enum_ov = len(transcript_tokens & enum_tokens)
        if enum_ov > 0:
            score += 0.25 * (1.0 + math.log(1 + enum_ov))

    return score, name_ov, enum_ov


# -----------------------------
# Semantic scoring (optional)
# -----------------------------

class ConceptNameEmbedIndex:
    def __init__(
        self,
        schema: SchemaIndex,
        embedder: Optional[SentenceTransformerEmbedder] = None,
    ) -> None:
        self.schema = schema
        self.embedder = embedder or SentenceTransformerEmbedder()
        self._concept_ids: List[str] = []
        self._concept_names: List[str] = []
        self._emb: Optional[np.ndarray] = None

    def build(self, batch_size: int = 64) -> None:
        self._concept_ids = self.schema.all_ids()
        self._concept_names = [normalize_text(self.schema.get(cid).name) for cid in self._concept_ids]
        self._emb = self.embedder.encode(self._concept_names, batch_size=batch_size)

    def _check(self) -> None:
        if self._emb is None:
            raise RuntimeError("ConceptNameEmbedIndex not built. Call build() first.")

    def top_m(self, query_text: str, m: int = 25) -> List[Tuple[str, float]]:
        self._check()
        q = normalize_text(query_text)
        q_emb = self.embedder.encode([q], batch_size=1)[0]
        sims = np.dot(self._emb, q_emb)
        if m >= len(sims):
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, m)[:m]
            idx = idx[np.argsort(-sims[idx])]
        out = [(self._concept_ids[i], float(sims[i])) for i in idx[:m]]
        return out


# -----------------------------
# Candidate generation
# -----------------------------

def generate_candidate_concepts(
    transcript: str,
    schema: SchemaIndex,
    retrieved_examples: Optional[List[RetrievalItem]] = None,
    *,
    # Target sizing
    target_size: int = 35,
    min_size: int = 20,
    max_size: int = 50,
    # Lexical config
    lexical_top_n: int = 40,
    lexical_min_score: float = 1.0,
    # Semantic config
    concept_embed_index: Optional[ConceptNameEmbedIndex] = None,
    semantic_top_m: int = 25,
    semantic_min_sim: float = 0.35,
    # Retrieval expansion
    add_retrieval_concepts: bool = True,
    # Common concept injection
    add_common_concepts: bool = True,
    # NEW: Must-include IDs (best recall boost with controlled risk)
    force_include_ids: Optional[Set[str]] = None,
) -> CandidateResult:
    reasons: Dict[str, List[CandidateReason]] = {}

    def add_reason(cid: str, reason: str, score: float) -> None:
        reasons.setdefault(cid, []).append(CandidateReason(concept_id=cid, reason=reason, score=score))

    t_text = normalize_text(transcript)
    t_tokens = set(meaningful_tokens(tokenize(t_text)))

    # A) Lexical matching against schema
    lex_scored: List[Tuple[str, float]] = []
    for cid in schema.all_ids():
        c = schema.get(cid)
        score, name_ov, enum_ov = lexical_score_concept(t_tokens, c)
        if score >= lexical_min_score:
            lex_scored.append((cid, score))
            add_reason(cid, f"lexical(score={score:.3f}, name_ov={name_ov}, enum_ov={enum_ov})", score)

    lex_scored.sort(key=lambda x: x[1], reverse=True)
    lex_ids = [cid for cid, sc in lex_scored[:lexical_top_n]]

    # B) Semantic matching (optional)
    sem_ids: List[str] = []
    if concept_embed_index is not None:
        sem_scored = concept_embed_index.top_m(t_text, m=semantic_top_m)
        for cid, sim in sem_scored:
            if sim >= semantic_min_sim:
                sem_ids.append(cid)
                add_reason(cid, f"semantic(sim={sim:.3f})", sim)

    # C) Retrieval expansion (concept ids present in retrieved examples)
    ret_ids: List[str] = []
    if add_retrieval_concepts and retrieved_examples:
        for r in retrieved_examples:
            for iv in r.gold_idvalues:
                cid = iv.id
                if schema.has_id(cid):
                    ret_ids.append(cid)
                    add_reason(cid, f"retrieval(example_id={r.example_id}, score={r.score:.3f})", r.score)

    # D) Common concepts by name pattern (optional)
    common_ids: Set[str] = set()
    if add_common_concepts:
        common_ids = default_common_concept_ids(schema)
        for cid in common_ids:
            add_reason(cid, "common_concept_by_name_pattern", 0.10)

    # E) NEW: Force include (high-FN) concept IDs
    if force_include_ids is None:
        force_include_ids = default_force_include_ids(schema)
    else:
        # keep only valid ids
        force_include_ids = {cid for cid in force_include_ids if schema.has_id(cid)}

    for cid in force_include_ids:
        add_reason(cid, "force_include_high_FN", 10.0)  # high score so it survives trimming

    # Combine (priority: lexical then semantic then retrieval then common)
    combined: List[str] = []
    for src in (lex_ids, sem_ids, ret_ids, list(common_ids)):
        for cid in src:
            if cid not in combined:
                combined.append(cid)

    candidate_ids = set(combined[:max_size])

    # If too few, pad with more lexical ids
    if len(candidate_ids) < min_size:
        for cid, sc in lex_scored[lexical_top_n:]:
            if cid not in candidate_ids:
                candidate_ids.add(cid)
                add_reason(cid, f"lexical_pad(score={sc:.3f})", sc)
            if len(candidate_ids) >= min_size:
                break

    # If still too few and semantic exists, pad from semantic
    if len(candidate_ids) < min_size and concept_embed_index is not None:
        sem_scored = concept_embed_index.top_m(t_text, m=max(semantic_top_m, min_size * 2))
        for cid, sim in sem_scored:
            if sim >= (semantic_min_sim * 0.9) and cid not in candidate_ids:
                candidate_ids.add(cid)
                add_reason(cid, f"semantic_pad(sim={sim:.3f})", sim)
            if len(candidate_ids) >= min_size:
                break

    # NEW: Inject force-included ids AFTER padding (so they always exist)
    candidate_ids |= set(force_include_ids)

    # If too many, trim by best reason score BUT keep force-included ids
    if len(candidate_ids) > max_size:
        cid_scores: List[Tuple[str, float]] = []
        for cid in candidate_ids:
            best = max((r.score for r in reasons.get(cid, [])), default=0.0)
            cid_scores.append((cid, best))
        cid_scores.sort(key=lambda x: x[1], reverse=True)

        kept: List[str] = []
        forced = set(force_include_ids)

        # keep all forced first
        for cid in sorted(forced, key=lambda x: int(x) if x.isdigit() else x):
            if cid in candidate_ids and cid not in kept:
                kept.append(cid)

        # fill remaining slots by score
        for cid, _ in cid_scores:
            if cid in kept:
                continue
            kept.append(cid)
            if len(kept) >= max_size:
                break

        candidate_ids = set(kept)

    # Create candidate table
    candidates: List[CandidateConcept] = []
    for cid in sorted(candidate_ids, key=lambda x: int(x) if x.isdigit() else x):
        c = schema.get(cid)
        candidates.append(
            CandidateConcept(
                id=c.id,
                name=c.name,
                value_type=c.value_type,
                value_enum=c.value_enum,
            )
        )

    return CandidateResult(candidate_ids=candidate_ids, candidates=candidates, reasons=reasons)
