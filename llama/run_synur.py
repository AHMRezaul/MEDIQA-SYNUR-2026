import argparse
import json
import os
from typing import Dict, List

from end_to_end import EndToEndRunner


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_queries_from_reference_jsonl(reference_jsonl: str) -> List[Dict[str, str]]:
    """
    Reference JSONL lines contain: {"id": ..., "transcript": ..., "observations": ...}
    We only need id + transcript as input for inference.
    """
    rows = read_jsonl(reference_jsonl)
    queries = []
    for r in rows:
        qid = str(r.get("id", "")).strip()
        transcript = str(r.get("transcript", "")).strip()
        if not qid or not transcript:
            continue
        queries.append({"id": qid, "transcript": transcript})
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True, help="Path to synur_schema.json")
    ap.add_argument("--train", required=True, help="Path to train.jsonl (KB)")
    ap.add_argument("--input_jsonl", required=True, help="JSONL with id+transcript (can be reference/dev/test file)")
    ap.add_argument("--out_dir", required=True, help="Output directory (will contain predictions.jsonl)")
    ap.add_argument("--cache_dir", default="../.cache")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llama_model", default="meta-llama/Llama-4-Scout-17B-16E-Instruct")

    # Retrieval / prompting / generation knobs
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--candidate_pool", type=int, default=30)
    ap.add_argument("--target_candidate_size", type=int, default=35)
    ap.add_argument("--max_exemplars", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--write_debug", action="store_true")
    args = ap.parse_args()

    # Build queries from input_jsonl
    queries = extract_queries_from_reference_jsonl(args.input_jsonl)

    runner = EndToEndRunner(
        schema_path=args.schema,
        train_jsonl_path=args.train,
        cache_dir=args.cache_dir,
        embed_model_name=args.embed_model,
        llama_model_id=args.llama_model,
    )

    run_info = runner.run_many(
        queries,
        out_dir=args.out_dir,
        write_debug_per_case=args.write_debug,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
        target_candidate_size=args.target_candidate_size,
        max_exemplars_in_prompt=args.max_exemplars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("Wrote:", run_info["predictions_jsonl"])
    print("Index:", os.path.join(args.out_dir, "_index.json"))


if __name__ == "__main__":
    main()
