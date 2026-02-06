"""
Step 9: End-to-end runner for research experiments + evaluator-ready output

This version:
- Runs the full pipeline for each query transcript:
    1) Retrieve top-K exemplars (Step 4)
    2) Generate candidate concept set (Step 5)
    3) Build prompt/messages (Step 6)
    4) Run Llama 4 inference (Step 7)
    5) Parse/repair + schema validate/normalize + expand (Step 8)
- Writes:
    (A) predictions.jsonl  (EVALUATOR-READY)
        Each line is:
          {"id": "<case_id>", "observations": [ {"id":..., "value_type":..., "value":...}, ... ]}
        (name is optional; scorer ignores it.)
    (B) _index.json summary
    (C) Optional per-case debug JSON files
"""

from __future__ import annotations
import sys
from tqdm import tqdm

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from define import SchemaIndex
from preprocess import build_kb_from_train_jsonl
from gatekeeper import summarize_schema
from rag import HybridConceptAwareRetriever, SentenceTransformerEmbedder, RetrievalItem
from schema_gen import ConceptNameEmbedIndex, generate_candidate_concepts
from prompt_builder import build_messages_for_llama4
from llama_inference import Llama4Config, Llama4Inference
from postprocess import postprocess_prediction, to_eval_jsonl_record, PostprocessOutput, postprocess_prediction_with_units


# -----------------------------
# File utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def write_json(path: str, obj: Any) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------
# Debug serialization helpers
# -----------------------------

def retrieval_item_to_dict(r: RetrievalItem) -> Dict[str, Any]:
    return {
        "example_id": r.example_id,
        "score": r.score,
        "s_transcript": r.s_transcript,
        "s_concept": r.s_concept,
        "s_lexical": r.s_lexical,
        "transcript": r.transcript,
        "gold_idvalues": [{"id": iv.id, "value": iv.value} for iv in r.gold_idvalues],
    }

def postprocess_to_debug_dict(p: PostprocessOutput) -> Dict[str, Any]:
    return {
        "parse_ok": p.parse_ok,
        "parse_error": p.parse_error,
        "raw_text": p.raw_text,
        "repaired_text_used": p.repaired_text_used,
        "parsed_idvalues": [{"id": iv.id, "value": iv.value} for iv in p.parsed_idvalues],
        "cleaned_idvalues": [{"id": iv.id, "value": iv.value} for iv in p.validation.cleaned],
        "dropped": [asdict(x) for x in p.validation.dropped],
        "expanded_observations": [
            {"id": o.id, "name": o.name, "value_type": o.value_type, "value": o.value}
            for o in p.expanded_observations
        ],
    }


# -----------------------------
# End-to-end runner
# -----------------------------

class EndToEndRunner:
    def __init__(
        self,
        *,
        schema_path: str = "../data/synur_schema.json",
        train_jsonl_path: str = "../data/train.jsonl",
        cache_dir: str = "../../../scratch/akarim9/.cache",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llama_model_id: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        llama_max_input_length: int = 8192,
        llama_default_max_new_tokens: int = 512,
        llama_default_temperature: float = 0.2,
        llama_default_top_p: float = 0.9,
        retrieval_weights: tuple[float, float, float] = (0.70, 0.25, 0.05),
    ) -> None:
        # ---- Schema
        self.schema_path = schema_path
        self.schema = SchemaIndex.from_json_file(schema_path)
        self.schema_summary = summarize_schema(self.schema)
        
        # ---- KB
        self.train_jsonl_path = train_jsonl_path
        self.kb, self.kb_stats = build_kb_from_train_jsonl(train_jsonl_path, schema=self.schema)

        # ---- Embedder shared across retrieval + concept semantic matching
        self.embedder = SentenceTransformerEmbedder(embed_model_name)

        # ---- Retriever (concept-aware hybrid)
        self.retriever = HybridConceptAwareRetriever(
            self.kb,
            schema=self.schema,
            embedder=self.embedder,
            weights=retrieval_weights,
        )
        self.retriever.build()

        # ---- Concept name semantic index (Step 5 semantic component)
        self.concept_index = ConceptNameEmbedIndex(self.schema, embedder=self.embedder)
        self.concept_index.build()

        # ---- Llama4 runner
        self.llm = Llama4Inference(
            Llama4Config(
                model_id=llama_model_id,
                cache_dir=cache_dir,
                max_input_length=llama_max_input_length,
                max_new_tokens=llama_default_max_new_tokens,
                temperature=llama_default_temperature,
                top_p=llama_default_top_p,
            )
        )
        self.llm.load()

    def run_one(
        self,
        *,
        case_id: str,
        transcript: str,
        out_dir: str,
        predictions_jsonl_path: str,
        include_name_in_predictions: bool = False,
        write_debug_per_case: bool = True,
        # Retrieval params
        top_k: int = 3,
        candidate_pool: int = 30,
        # Candidate set params
        target_candidate_size: int = 35,
        # Prompt params
        max_exemplars_in_prompt: int = 3,
        # Generation params
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Run pipeline for a single case and append evaluator-ready line to predictions_jsonl_path.
        Returns a compact per-case summary for the index file.
        """
        case_id = str(case_id)
        transcript = str(transcript)

        # 1) Retrieve exemplars
        retrieved = self.retriever.retrieve(
            transcript,
            k=top_k,
            candidate_pool=candidate_pool,
            add_schema_hints_to_query=True,
            max_schema_hints=30,
        )

        # 2) Candidate concept generation
        cand = generate_candidate_concepts(
            transcript=transcript,
            schema=self.schema,
            retrieved_examples=retrieved,
            concept_embed_index=self.concept_index,
            target_size=target_candidate_size,
        )

        # 3) Prompt construction
        messages = build_messages_for_llama4(
            query_transcript=transcript,
            retrieved_examples=retrieved,
            candidate_result=cand,
            max_exemplars=max_exemplars_in_prompt,
            include_candidate_table=True,
            include_exemplar_outputs=True,
        )

        # 4) LLM inference
        raw_text = self.llm.generate_from_messages(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # 5) Postprocess: parse/repair + schema validate/normalize + expand
        post = postprocess_prediction_with_units(
            raw_text=raw_text,
            transcript=transcript,
            schema=self.schema,
            candidate_ids=cand.candidate_ids,
            drop_none_values=True,
        )

        # 6) Write evaluator-ready prediction JSONL line
        eval_record = to_eval_jsonl_record(
            example_id=case_id,
            post=post,
            include_name=include_name_in_predictions,
        )
        append_jsonl(predictions_jsonl_path, eval_record)

        # 7) Optional debug artifact
        if write_debug_per_case:
            debug_obj: Dict[str, Any] = {
                "case_id": case_id,
                "transcript": transcript,
                "schema_summary": asdict(self.schema_summary),
                "kb_stats": asdict(self.kb_stats),
                "retrieval": [retrieval_item_to_dict(r) for r in retrieved],
                "candidate_ids": sorted(list(cand.candidate_ids), key=lambda x: int(x) if x.isdigit() else x),
                "candidate_table": [
                    {"id": c.id, "name": c.name, "value_type": c.value_type, "value_enum": c.value_enum}
                    for c in cand.candidates
                ],
                "candidate_reasons": {
                    cid: [asdict(rr) for rr in rs]
                    for cid, rs in cand.reasons.items()
                    if cid in cand.candidate_ids
                },
                "raw_model_output": raw_text,
                "postprocess": postprocess_to_debug_dict(post),
                "eval_record_written": eval_record,
            }
            write_json(os.path.join(out_dir, f"{case_id}.debug.json"), debug_obj)

        # Return compact summary for index
        return {
            "id": case_id,
            "parse_ok": post.parse_ok,
            "num_cleaned": len(post.validation.cleaned),
            "num_dropped": len(post.validation.dropped),
        }

    def run_many(
        self,
        queries: Sequence[Dict[str, str]],
        *,
        out_dir: str = "runs/run1",
        predictions_jsonl_name: str = "predictions.jsonl",
        include_name_in_predictions: bool = False,
        write_debug_per_case: bool = True,
        # Retrieval params
        top_k: int = 3,
        candidate_pool: int = 30,
        # Candidate set params
        target_candidate_size: int = 35,
        # Prompt params
        max_exemplars_in_prompt: int = 3,
        # Generation params
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Run pipeline for all queries and write:
          - predictions.jsonl (evaluator-ready)
          - _index.json summary
          - optional per-case debug files
        """
        os.makedirs(out_dir, exist_ok=True)

        predictions_path = os.path.join(out_dir, predictions_jsonl_name)

        # Reset predictions file for this run
        if os.path.exists(predictions_path):
            os.remove(predictions_path)

        index: List[Dict[str, Any]] = []

        # simple running stats
        n_ok = 0
        total_cleaned = 0
        total_dropped = 0

        pbar = tqdm(
            queries,
            desc="Generating",
            total=len(queries),
            unit="case",
            file=sys.stderr,      # more reliable on Slurm
            dynamic_ncols=True,
        )

        for i, q in enumerate(pbar):
            case_id = str(q.get("id", "")).strip() or f"q{i}"
            transcript = str(q.get("transcript", "")).strip()

            summary = self.run_one(
                case_id=case_id,
                transcript=transcript,
                out_dir=out_dir,
                predictions_jsonl_path=predictions_path,
                include_name_in_predictions=include_name_in_predictions,
                write_debug_per_case=write_debug_per_case,
                top_k=top_k,
                candidate_pool=candidate_pool,
                target_candidate_size=target_candidate_size,
                max_exemplars_in_prompt=max_exemplars_in_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            index.append(summary)

            # update stats for postfix
            if summary["parse_ok"]:
                n_ok += 1
            total_cleaned += int(summary["num_cleaned"])
            total_dropped += int(summary["num_dropped"])

            done = i + 1
            pbar.set_postfix(
                parse_ok=f"{n_ok}/{done}",
                avg_cleaned=f"{(total_cleaned / done):.2f}",
                avg_dropped=f"{(total_dropped / done):.2f}",
            )

        run_index = {
            "predictions_jsonl": predictions_path,
            "num_queries": len(index),
            "index": index,
        }
        write_json(os.path.join(out_dir, "_index.json"), run_index)
        return run_index