# MEDIQA-SYNUR-2026

This repository contains code for **MEDIQA-SYNUR @ ClinicalNLP 2026: Observation Extraction from Nurse Dictations**. The task focuses on **extracting and normalizing clinical observations from nurse dictations / nurse–patient conversation transcripts and mapping them to a large ontology of clinical concepts**, with the goal of reducing documentation burden and improving record accuracy/completeness. Task page: [https://sites.google.com/view/mediqa2026/mediqa-synur](https://sites.google.com/view/mediqa2026/mediqa-synur) ([Google Sites][1])

Repo: [https://github.com/AHMRezaul/MEDIQA-SYNUR-2026](https://github.com/AHMRezaul/MEDIQA-SYNUR-2026)

## Two approaches in this repo

1. `llama/`: Llama (Llama 4 Scout) inference + RAG using a MiniLM embedder (FAISS/RAG is handled internally by the script).
2. `gpt/`: OpenAI API inference (GPT-5.2) + RAG using OpenAI `text-embedding-3-large` (handled internally by the script), plus a best result with second-pass reranking/selection script.

## Approach 1: Llama (Llama 4 Scout) — run `run_synur.py`

Command:
```
python run_synur.py 
--schema "$SCHEMA" 
--train "$TRAIN" 
--input_jsonl "$INPUT" 
--out_dir "$OUTDIR" 
--top_k 3 
--candidate_pool 30 
--target_candidate_size 35 
--max_exemplars 3 
--max_new_tokens 512 
--temperature 0.2 
--top_p 0.9
```
## Approach 2: GPT-5.2 + text-embedding-3-large — run `run_synur.py`

Command:
```
python run_synur.py 
--schema "$SCHEMA" 
--train "$TRAIN" 
--input_jsonl "$INPUT" 
--out_dir "$OUTDIR" 
--embed_model text-embedding-3-large 
--llm_model gpt-5.2 
--top_k 3 
--candidate_pool 30 
--target_candidate_size 35 
--max_exemplars 3 
--max_new_tokens 512 
--temperature 0.2 
--top_p 0.9
```
## GPT second pass (Final Output File): `run_synur_secondpass.py`

Command:
```
python run_synur_secondpass.py 
--train data/database.jsonl 
--dev data/test.jsonl 
--schema data/synur_schema.json 
--out runs/run_${SLURM_JOB_ID}/pred.jsonl 
--use_mmr 
--top_k 50
```
[1]: https://sites.google.com/view/mediqa2026/mediqa-synur "MEDIQA 2026 - MEDIQA-SYNUR"
