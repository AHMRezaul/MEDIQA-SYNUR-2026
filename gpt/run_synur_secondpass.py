"""
MEDIQA-SYNUR RAG-based Observation Extraction System (improved retrieval + 2-pass audit)
"""

import argparse
import hashlib
import json
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def backoff_sleep(attempt: int, base: float = 0.8, cap: float = 15.0) -> None:
    time.sleep(min(cap, base * (2 ** attempt)))


def safe_json_loads_maybe(x: Any) -> Any:
    """Train 'observations' may be a JSON string; return list if possible."""
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            return json.loads(s)
        except Exception:
            return []
    return x


def extract_json_array(text: str) -> List[Any]:
    """Best-effort JSON array extractor from model output."""
    t = (text or "").strip()

    # Remove markdown fences if present
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # Direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # Find a [...]-block
    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end != -1 and end > start:
        chunk = t[start:end + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    return []


def safe_id_set(obs: Any) -> set:
    """Return set of ids present in observation list."""
    if not isinstance(obs, list):
        return set()
    out = set()
    for o in obs:
        if isinstance(o, dict) and "id" in o:
            out.add(str(o["id"]))
    return out


# -----------------------------
# Main RAG system
# -----------------------------

class MEDIQASYNURRagImproved:
    def __init__(
        self,
        train_file: str,
        schema_file: str,
        dev_file: str,
        output_file: str,
        embed_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-5.2",
        top_k: int = 50,
        embed_batch_size: int = 64,
        embed_dims: Optional[int] = None,
        cache_dir: str = "cache_synur",
        use_mmr: bool = False,
        mmr_pool: int = 100,
        mmr_lambda: float = 0.85,
        temperature: float = 0.0,
        max_completion_tokens: int = 4096,
        # 2nd pass controls
        enable_audit: bool = True,
        audit_guardrail_added_max: int = 8,  # if pass2 adds > this many new IDs, fallback to pass1
    ):
        self.train_file = train_file
        self.schema_file = schema_file
        self.dev_file = dev_file
        self.output_file = output_file

        self.embed_model = embed_model
        self.embed_dims = embed_dims
        self.embed_batch_size = embed_batch_size
        self.cache_dir = cache_dir

        self.llm_model = llm_model
        self.top_k = top_k
        self.use_mmr = use_mmr
        self.mmr_pool = mmr_pool
        self.mmr_lambda = mmr_lambda
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

        self.enable_audit = enable_audit
        self.audit_guardrail_added_max = audit_guardrail_added_max

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        self.train_data: List[Dict[str, Any]] = []
        self.dev_data: List[Dict[str, Any]] = []
        self.schema: List[Dict[str, Any]] = []

        # Embeddings matrix: normalized float32 [N, d]
        self.E: Optional[np.ndarray] = None

        print("Loading data...")
        self.load_data()

        print("Loading/building cached embeddings...")
        self.load_or_build_train_embeddings()

    def load_data(self):
        self.train_data = read_jsonl(self.train_file)
        print(f"Loaded {len(self.train_data)} training examples")

        with open(self.schema_file, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        print(f"Loaded schema with {len(self.schema)} observation types")

        self.dev_data = read_jsonl(self.dev_file)
        print(f"Loaded {len(self.dev_data)} examples")

    # -----------------------------
    # Embeddings (batched + cached)
    # -----------------------------

    def _cache_paths(self) -> Tuple[str, str]:
        os.makedirs(self.cache_dir, exist_ok=True)
        fp = sha1_of_file(self.train_file)
        base = f"train_emb_{fp}_{self.embed_model}"
        if self.embed_dims is not None:
            base += f"_d{self.embed_dims}"
        emb_path = os.path.join(self.cache_dir, base + ".npy")
        meta_path = os.path.join(self.cache_dir, base + ".meta.json")
        return emb_path, meta_path

    def get_embeddings_batched(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in tqdm(range(0, len(texts), self.embed_batch_size), desc="Embedding train", unit="batch"):
            batch = texts[i : i + self.embed_batch_size]
            for attempt in range(8):
                try:
                    kwargs = {"model": self.embed_model, "input": batch}
                    if self.embed_dims is not None:
                        kwargs["dimensions"] = self.embed_dims
                    resp = self.client.embeddings.create(**kwargs)
                    out.extend([d.embedding for d in resp.data])
                    break
                except Exception:
                    if attempt == 7:
                        raise
                    backoff_sleep(attempt)
        return out

    def get_embedding_one(self, text: str) -> np.ndarray:
        for attempt in range(8):
            try:
                kwargs = {"model": self.embed_model, "input": text}
                if self.embed_dims is not None:
                    kwargs["dimensions"] = self.embed_dims
                resp = self.client.embeddings.create(**kwargs)
                v = np.array(resp.data[0].embedding, dtype=np.float32)
                v /= (np.linalg.norm(v) + 1e-12)
                return v
            except Exception:
                if attempt == 7:
                    raise
                backoff_sleep(attempt)
        raise RuntimeError("unreachable")

    def load_or_build_train_embeddings(self):
        emb_path, meta_path = self._cache_paths()

        if os.path.exists(emb_path) and os.path.exists(meta_path):
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            if meta.get("n") == len(self.train_data):
                E = np.load(emb_path).astype(np.float32, copy=False)
                self.E = E
                print(f"Loaded cached embeddings: {emb_path}")
                return

        texts = [ex["transcript"] for ex in self.train_data]
        embs = self.get_embeddings_batched(texts)
        E = np.array(embs, dtype=np.float32)

        # Normalize for cosine similarity
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        self.E = E

        np.save(emb_path, E)
        json.dump(
            {"n": len(self.train_data), "model": self.embed_model, "dims": self.embed_dims},
            open(meta_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        print(f"Saved cached embeddings: {emb_path}")

    # -----------------------------
    # Retrieval (fast + optional MMR)
    # -----------------------------

    def retrieve_similar_examples(self, query_transcript: str) -> List[Dict[str, Any]]:
        assert self.E is not None
        q = self.get_embedding_one(query_transcript)

        if self.use_mmr:
            idxs = self._topk_mmr(q, k=self.top_k, pool=self.mmr_pool, lam=self.mmr_lambda)
        else:
            idxs = self._topk(q, k=self.top_k)

        return [self.train_data[i] for i in idxs]

    def _topk(self, q: np.ndarray, k: int) -> List[int]:
        sims = self.E @ q  # [N]
        k = min(k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [int(i) for i in idx]

    def _topk_mmr(self, q: np.ndarray, k: int, pool: int = 50, lam: float = 0.85) -> List[int]:
        sims = self.E @ q
        pool = min(pool, sims.shape[0])
        cand = np.argpartition(-sims, pool - 1)[:pool]
        cand = cand[np.argsort(-sims[cand])]
        cand_vecs = self.E[cand]
        sim_q = cand_vecs @ q

        selected_local: List[int] = []
        for _ in range(min(k, pool)):
            best_j = None
            best_score = -1e9
            for j in range(pool):
                if j in selected_local:
                    continue
                if not selected_local:
                    score = float(sim_q[j])
                else:
                    max_sim_to_sel = float(np.max(cand_vecs[selected_local] @ cand_vecs[j]))
                    score = lam * float(sim_q[j]) - (1.0 - lam) * max_sim_to_sel
                if score > best_score:
                    best_score = score
                    best_j = j
            selected_local.append(best_j)

        return [int(cand[j]) for j in selected_local]

    # -----------------------------
    # Prompting (unchanged style)
    # -----------------------------

    def format_schema_for_prompt(self) -> str:
        schema_text = "Available Observation Types:\n\n"
        for obs in self.schema:
            schema_text += f"ID: {obs['id']}\n"
            schema_text += f"Name: {obs['name']}\n"
            schema_text += f"Value Type: {obs['value_type']}\n"
            if "value_enum" in obs:
                schema_text += f"Valid Values: {', '.join(map(str, obs['value_enum']))}\n"
            schema_text += "\n"
        return schema_text

    def format_examples_for_prompt(self, examples: List[Dict[str, Any]]) -> str:
        examples_text = "Here are similar transcripts with their extracted observations:\n\n"
        for i, ex in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Transcript: {ex['transcript']}\n"
            examples_text += f"Observations: {ex['observations']}\n\n"
        return examples_text

    def create_prompt(self, transcript: str, similar_examples: List[Dict[str, Any]]) -> str:
        schema_info = self.format_schema_for_prompt()
        examples_info = self.format_examples_for_prompt(similar_examples)

        prompt = f"""You are a clinical observation extraction system for nursing documentation. Your task is to extract structured clinical observations from nurse dictation transcripts.

{examples_info}

Now, extract observations from the following transcript: 

Transcript:
{transcript}

Follow these rules:
1. Only extract observations that are explicitly mentioned or strongly implied in the transcript
2. Use the exact observation IDs, names, and value types from the schema
3. For SINGLE_SELECT: choose ONE value from the valid values list
4. For MULTI_SELECT: choose one or more values from the valid values list as an array
5. For NUMERIC: extract the numeric value only
6. For STRING: extract the relevant text information
7. Output ONLY a valid JSON array of observations (no explanation, no markdown)

{schema_info}

Output the observations as a JSON array in this exact format:
[{{"id": "...", "value_type": "...", "name": "...", "value": ...}}, ...]

IMPORTANT: Output ONLY the JSON array, nothing else."""
        return prompt

    # -----------------------------
    # Second pass: Audit and refine
    # -----------------------------

    def audit_and_refine(
        self,
        transcript: str,
        pass1_obs: List[Dict[str, Any]],
        similar_examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        schema_info = self.format_schema_for_prompt()
        examples_info = self.format_examples_for_prompt(similar_examples)

        pass1_json = json.dumps(pass1_obs, ensure_ascii=False)

        already_ids = sorted(list(safe_id_set(pass1_obs)))
        already_ids_text = ", ".join(already_ids) if already_ids else "(none)"

        audit_prompt = f"""You are an auditor for a clinical observation extraction system.

Your goal: Only add an observation if it is clearly supported by the transcript (explicitly mentioned or strongly implied).
If you are uncertain, DO NOT add.

{schema_info}

{examples_info}

Transcript:
{transcript}

Model's current extracted observations (JSON array):
{pass1_json}

Observation IDs already present:
{already_ids_text}

Audit instructions:
1) First, VERIFY the current extracted observations:
   - Remove any observation that is NOT explicitly mentioned or strongly implied by the transcript.
   - For SINGLE_SELECT and MULTI_SELECT, ensure the value(s) match the schema enum AND best match the transcript wording.
   - For NUMERIC, ensure the value is a number only (no units).
2) When auditing for omissions, pay extra attention to: cognitive status, dyspnea, mobility, GI symptoms, urine appearance, patient safety, pulse, and secondary diagnosis — ONLY if mentioned.
3) Then, look for HIGH-CONFIDENCE missing observations that are clearly supported by the transcript and add them.
4) Unit-only observations rule:
   - Only output unit fields (e.g., oxygen saturation unit, bladder scan volume unit) if the transcript includes the corresponding numeric measurement.
   - If no measurement is present, do NOT include the unit observation.
5) Keep the output format identical (JSON array of observation objects).
6) Output ONLY the final JSON array (no explanation, no markdown).


Return the final observations as a JSON array:
[{{"id": "...", "value_type": "...", "name": "...", "value": ...}}, ...]
IMPORTANT: Output ONLY the JSON array."""

        raw = ""
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "Return only a valid JSON array. No explanations."},
                        {"role": "user", "content": audit_prompt},
                    ],
                    temperature=0.0,
                    max_completion_tokens=self.max_completion_tokens,
                )
                raw = (resp.choices[0].message.content or "").strip()
                break
            except Exception:
                if attempt == 5:
                    raw = ""
                else:
                    backoff_sleep(attempt)

        refined = extract_json_array(raw)
        if not refined:
            return pass1_obs

        # Guardrail: if audit adds too many new IDs, likely hallucination -> keep pass1
        pass1_ids = safe_id_set(pass1_obs)
        ref_ids = safe_id_set(refined)
        added = len(ref_ids - pass1_ids)
        if self.audit_guardrail_added_max is not None and added > self.audit_guardrail_added_max:
            return pass1_obs

        return refined

    # -----------------------------
    # LLM call + output formatting
    # -----------------------------

    def extract_observations(self, transcript: str) -> List[Dict[str, Any]]:
        similar_examples = self.retrieve_similar_examples(transcript)
        prompt = self.create_prompt(transcript, similar_examples)

        raw = ""
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a clinical observation extraction assistant. Output only valid JSON, no explanations."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                )
                raw = (resp.choices[0].message.content or "").strip()
                break
            except Exception:
                if attempt == 5:
                    raw = ""
                else:
                    backoff_sleep(attempt)

        pass1_obs = extract_json_array(raw)
        if not isinstance(pass1_obs, list):
            pass1_obs = []

        if not self.enable_audit:
            return pass1_obs

        pass2_obs = self.audit_and_refine(
            transcript=transcript,
            pass1_obs=pass1_obs,
            similar_examples=similar_examples,
        )
        return pass2_obs if isinstance(pass2_obs, list) else pass1_obs

    def run_inference(self):
        print(f"\nRunning inference on {len(self.dev_data)} examples...")

        results = []
        for example in tqdm(self.dev_data, desc="Processing examples", unit="example"):
            sid = example["id"]
            transcript = example["transcript"]

            try:
                observations = self.extract_observations(transcript)
                results.append({"id": sid, "observations": observations})
            except Exception as e:
                tqdm.write(f"ERROR processing example {sid}: {e}")
                results.append({"id": sid, "observations": []})

        print(f"\nSaving results to {self.output_file}...")
        write_jsonl(self.output_file, results)
        print(f"Done! Predictions saved to {self.output_file}")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--use_mmr", action="store_true")
    ap.add_argument("--mmr_pool", type=int, default=100)
    ap.add_argument("--mmr_lambda", type=float, default=0.85)

    ap.add_argument("--embed_model", default="text-embedding-3-large")
    ap.add_argument("--embed_batch_size", type=int, default=64)
    ap.add_argument("--embed_dims", type=int, default=None)
    ap.add_argument("--cache_dir", default="cache_synur")

    ap.add_argument("--llm_model", default="gpt-5.2")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_completion_tokens", type=int, default=4096)

    # Second-pass controls
    ap.add_argument("--no_audit", action="store_true", help="Disable second-pass audit/refine")
    ap.add_argument("--audit_guardrail_added_max", type=int, default=8,
                    help="If audit adds more than this many new IDs, fallback to pass1")

    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set!")

    rag = MEDIQASYNURRagImproved(
        train_file=args.train,
        schema_file=args.schema,
        dev_file=args.dev,
        output_file=args.out,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
        embed_batch_size=args.embed_batch_size,
        embed_dims=args.embed_dims,
        cache_dir=args.cache_dir,
        use_mmr=args.use_mmr,
        mmr_pool=args.mmr_pool,
        mmr_lambda=args.mmr_lambda,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        enable_audit=(not args.no_audit),
        audit_guardrail_added_max=args.audit_guardrail_added_max,
    )

    rag.run_inference()


if __name__ == "__main__":
    main()
