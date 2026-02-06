from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from define import IDValue
from dataclasses import is_dataclass, asdict
import numpy as np


def _to_idvalue_list(x):
    """
    Convert metadata-loaded gold_idvalues into List[IDValue].
    Supports:
      - None
      - already List[IDValue]
      - list of dicts like {"id": "...", "value": ...}
    """
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, IDValue):
                out.append(item)
            elif isinstance(item, dict):
                cid = str(item.get("id", "")).strip()
                if not cid:
                    continue
                out.append(IDValue(id=cid, value=item.get("value")))
            else:
                # fallback: try attribute access
                cid = getattr(item, "id", None)
                val = getattr(item, "value", None)
                if cid is None:
                    continue
                out.append(IDValue(id=str(cid).strip(), value=val))
        return out
    # anything else -> empty
    return []


def _idvalue_list_to_jsonable(x):
    """
    Convert List[IDValue] (or list of objects with .id/.value) into JSONable list[dict].
    """
    if x is None:
        return None
    if not isinstance(x, list):
        return x
    out = []
    for item in x:
        if isinstance(item, dict):
            # already jsonable
            out.append({"id": str(item.get("id", "")).strip(), "value": item.get("value")})
        else:
            cid = getattr(item, "id", None)
            val = getattr(item, "value", None)
            if cid is None:
                continue
            out.append({"id": str(cid).strip(), "value": val})
    return out

def _to_jsonable(x):
    """
    Convert common Python objects (including dataclasses) into JSON-serializable structures.
    """
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # dataclasses (Observation is very likely a dataclass)
    if is_dataclass(x):
        return {k: _to_jsonable(v) for k, v in asdict(x).items()}

    # dict / list / tuple
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # objects with __dict__ (fallback)
    if hasattr(x, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(x).items()}

    # final fallback
    return str(x)



@dataclass(frozen=True)
class FaissStorePaths:
    index_transcript: str
    index_concept: str
    emb_transcript_npy: str
    emb_concept_npy: str
    meta_jsonl: str
    config_json: str


def _paths(index_dir: str) -> FaissStorePaths:
    return FaissStorePaths(
        index_transcript=os.path.join(index_dir, "index_transcript.faiss"),
        index_concept=os.path.join(index_dir, "index_concept.faiss"),
        emb_transcript_npy=os.path.join(index_dir, "emb_transcript.npy"),
        emb_concept_npy=os.path.join(index_dir, "emb_concept.npy"),
        meta_jsonl=os.path.join(index_dir, "kb_meta.jsonl"),
        config_json=os.path.join(index_dir, "config.json"),
    )



class PersistentFaissDualIndex:
    """
    Persistent dual FAISS index:
      - transcript embeddings index
      - concept_text embeddings index

    Also persists:
      - embedding matrices (optional but useful for reranking / diagnostics)
      - KB metadata (example_id, transcript, concept_text, gold fields)

    Uses IndexFlatIP (exact inner-product). With normalized embeddings, IP == cosine similarity.
    """

    def __init__(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self.p = _paths(index_dir)

        self.dim: Optional[int] = None
        self.n: Optional[int] = None

        self._faiss_t = None
        self._faiss_c = None

        # stored embeddings (loaded as memmap for efficiency)
        self._emb_t: Optional[np.ndarray] = None
        self._emb_c: Optional[np.ndarray] = None

        # loaded KB metadata arrays
        self.example_ids: List[str] = []
        self.transcripts: List[str] = []
        self.concept_texts: List[str] = []
        self.gold_idvalues: List[object] = []
        self.gold_observations: List[object] = []

    def exists(self) -> bool:
        return (
            os.path.exists(self.p.index_transcript)
            and os.path.exists(self.p.index_concept)
            and os.path.exists(self.p.meta_jsonl)
            and os.path.exists(self.p.config_json)
        )

    def build_and_save(
        self,
        *,
        transcript_emb: np.ndarray,
        concept_emb: np.ndarray,
        example_ids: List[str],
        transcripts: List[str],
        concept_texts: List[str],
        gold_idvalues: List[object],
        gold_observations: List[object],
        embed_model_name: str,
    ) -> None:
        """
        Build FAISS indices and save everything to disk.
        """
        import faiss  # local import

        os.makedirs(self.index_dir, exist_ok=True)

        if transcript_emb.dtype != np.float32:
            transcript_emb = transcript_emb.astype(np.float32)
        if concept_emb.dtype != np.float32:
            concept_emb = concept_emb.astype(np.float32)

        if transcript_emb.ndim != 2 or concept_emb.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays (N, d).")
        if transcript_emb.shape != concept_emb.shape:
            raise ValueError("Transcript and concept embeddings must have the same shape (N, d).")

        n, d = transcript_emb.shape
        if len(example_ids) != n:
            raise ValueError("example_ids length must match embeddings N.")

        self.n = n
        self.dim = d

        # Save embedding matrices for optional reranking/debug (memmap friendly).
        # NOTE: np.save stores a header; mmap_mode='r' works.
        np.save(self.p.emb_transcript_npy, transcript_emb)
        np.save(self.p.emb_concept_npy, concept_emb)

        # Build exact FAISS indices (inner product).
        idx_t = faiss.IndexFlatIP(d)
        idx_c = faiss.IndexFlatIP(d)
        idx_t.add(transcript_emb)
        idx_c.add(concept_emb)

        faiss.write_index(idx_t, self.p.index_transcript)
        faiss.write_index(idx_c, self.p.index_concept)

        # Save KB metadata (1 line per example).
        with open(self.p.meta_jsonl, "w", encoding="utf-8") as f:
            for i in range(n):
                rec = {
                    "example_id": example_ids[i],
                    "transcript": transcripts[i],
                    "concept_text": concept_texts[i],
                    "gold_idvalues": _idvalue_list_to_jsonable(gold_idvalues[i]),
                    "gold_observations": _to_jsonable(gold_observations[i]),    

                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Save config
        cfg = {
            "n": n,
            "dim": d,
            "embed_model_name": embed_model_name,
        }
        with open(self.p.config_json, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # Load into memory for immediate use
        self.load()

    def load(self) -> None:
        """
        Load indices + metadata from disk.
        """
        import faiss  # local import

        if not self.exists():
            raise FileNotFoundError(f"FAISS store not found in: {self.index_dir}")

        with open(self.p.config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.n = int(cfg["n"])
        self.dim = int(cfg["dim"])

        self._faiss_t = faiss.read_index(self.p.index_transcript)
        self._faiss_c = faiss.read_index(self.p.index_concept)

        # Memory-map embeddings (optional, but useful and cheap)
        self._emb_t = np.load(self.p.emb_transcript_npy, mmap_mode="r")
        self._emb_c = np.load(self.p.emb_concept_npy, mmap_mode="r")

        # Load KB metadata
        self.example_ids = []
        self.transcripts = []
        self.concept_texts = []
        self.gold_idvalues = []
        self.gold_observations = []

        with open(self.p.meta_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.example_ids.append(rec["example_id"])
                self.transcripts.append(rec["transcript"])
                self.concept_texts.append(rec["concept_text"])
                self.gold_idvalues.append(_to_idvalue_list(rec.get("gold_idvalues")))
                self.gold_observations.append(rec.get("gold_observations"))

        if len(self.example_ids) != self.n:
            raise RuntimeError("Loaded metadata size does not match config n.")

    def search_transcript(self, q_emb: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._faiss_t is None:
            raise RuntimeError("FAISS store not loaded.")
        q = q_emb.astype(np.float32, copy=False)
        if q.ndim == 1:
            q = q[None, :]
        scores, idx = self._faiss_t.search(q, topk)
        return scores[0], idx[0]

    def search_concept(self, q_emb: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._faiss_c is None:
            raise RuntimeError("FAISS store not loaded.")
        q = q_emb.astype(np.float32, copy=False)
        if q.ndim == 1:
            q = q[None, :]
        scores, idx = self._faiss_c.search(q, topk)
        return scores[0], idx[0]

    def get_emb_transcript(self) -> np.ndarray:
        if self._emb_t is None:
            raise RuntimeError("Embeddings not loaded.")
        return self._emb_t

    def get_emb_concept(self) -> np.ndarray:
        if self._emb_c is None:
            raise RuntimeError("Embeddings not loaded.")
        return self._emb_c

    def get_embed_model_name(self) -> Optional[str]:
        if not os.path.exists(self.p.config_json):
            return None
        with open(self.p.config_json, "r", encoding="utf-8") as f:
            return json.load(f).get("embed_model_name")
