"""
Step 6: Prompt/message construction for Llama 4 Instruct (chat template)

Goal:
- Build a single chat prompt that includes:
  (1) system instructions (strict JSON only, output is list of {id,value})
  (2) a compact candidate concept table (id, name, value_type, and enums for select-types)
  (3) 2–3 retrieved few-shot exemplars (transcript + gold output in canonical {id,value} form)
  (4) the query transcript to solve

Inputs:
- query_transcript: str
- retrieved_examples: List[RetrievalItem] (Step 4)
- candidate_result: CandidateResult (Step 5)
- schema: SchemaIndex (optional; not required if CandidateConcept already holds fields)

Output:
- messages: List[Dict] in the format expected by:
    processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

Notes:
- This step *only* builds the prompt/messages. No model invocation here.
- We keep the exemplar outputs as canonical [{id,value}, ...] JSON.
- We deliberately DO NOT ask the model to output name/value_type; those come from schema later.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from define import IDValue
from rag import RetrievalItem
from schema_gen import CandidateResult, CandidateConcept


# -----------------------------
# System instructions
# -----------------------------

# SYSTEM_MSG = (
#     "You are a clinical information extraction assistant.\n"
#     "Task: Extract structured flowsheet observations from a clinician transcript.\n"
#     "\n"
#     "OUTPUT FORMAT (STRICT):\n"
#     "- Return ONLY valid JSON.\n"
#     "- The JSON must be a list of objects.\n"
#     "- Each object must have exactly two keys: \"id\" and \"value\".\n"
#     "- Example:\n"
#     "  [{\"id\": \"38\", \"value\": \"Yes\"}, {\"id\": \"33\", \"value\": 98}]\n"
#     "\n"
#     "CONSTRAINTS:\n"
#     "- You MUST choose observation ids ONLY from the provided Candidate Concepts list.\n"
#     "- For SINGLE_SELECT or MULTI_SELECT concepts, \"value\" MUST use ONLY allowed enum values shown.\n"
#     "- For MULTI_SELECT, \"value\" MUST be a JSON list of strings (even if one item).\n"
#     "- For NUMERIC, \"value\" MUST be a number (no units or extra text).\n"
#     "- For STRING, \"value\" MUST be a string.\n"
#     "- If an observation is not stated or cannot be inferred, omit it.\n"
#     "- Do not output duplicate ids.\n"
#     "\n"
#     "IMPORTANT:\n"
#     "- Do NOT include any explanations.\n"
#     "- Do NOT wrap JSON in code fences.\n"
# )

SYSTEM_MSG = (
    "You are a clinical information extraction assistant.\n"
    "Task: Extract structured flowsheet observations from a clinician transcript.\n"
    "\n"
    "WORKFLOW (MANDATORY):\n"
    "1) Carefully scan the entire transcript for ANY explicitly stated flowsheet-relevant findings "
    "(e.g., vitals, symptoms, exam findings, mental status, mobility/safety, intake/output, pain, respiratory, GI/GU, interventions).\n"
    "2) Output every observation that is explicitly supported AND appears in the Candidate Concepts list.\n"
    "\n"
    "OUTPUT FORMAT (STRICT):\n"
    "- Return ONLY valid JSON.\n"
    "- Your entire response MUST start with '[' and end with ']'.\n"
    "- The JSON must be a list of objects.\n"
    "- Each object must have exactly two keys: \"id\" and \"value\".\n"
    "- Example:\n"
    "  [{\"id\": \"38\", \"value\": \"Yes\"}, {\"id\": \"33\", \"value\": 98}]\n"
    "\n"
    "CONSTRAINTS:\n"
    "- You MUST choose observation ids ONLY from the provided Candidate Concepts list.\n"
    "- For SINGLE_SELECT or MULTI_SELECT concepts, \"value\" MUST be exactly one of the allowed enum values shown.\n"
    "- For MULTI_SELECT, \"value\" MUST be a JSON list of strings (even if one item) and must include all applicable enum values for that id.\n"
    "- For NUMERIC, \"value\" MUST be a number only (no units, no extra text).\n"
    "- For STRING, \"value\" MUST be copied verbatim from the transcript (no paraphrasing, no abbreviation expansion).\n"
    "- If an observation is not explicitly stated in the transcript, omit it (do not guess or infer).\n"
    "- Do not output the same id in more than one object.\n"
    "\n"
    "IMPORTANT:\n"
    "- Do NOT include any explanations.\n"
    "- Do NOT wrap JSON in code fences.\n"
)


# -----------------------------
# Helpers: content blocks + formatting
# -----------------------------

def _txt(s: str) -> Dict[str, Any]:
    return {"type": "text", "text": s}

def idvalues_to_canonical_json(idvalues: Sequence[IDValue], indent: int = 2) -> str:
    payload = [{"id": iv.id, "value": iv.value} for iv in idvalues]
    return json.dumps(payload, ensure_ascii=False, indent=indent)

def format_candidate_table(
    candidates: Sequence[CandidateConcept],
    *,
    include_enum_for_select: bool = True,
    max_enum_items: int = 80,
) -> str:
    """
    Produce a compact plain-text candidate table for the prompt.
    We include enums for select types, truncated to max_enum_items to limit prompt length.
    """
    lines: List[str] = []
    lines.append("Candidate Concepts (you may ONLY use these ids):")
    for c in candidates:
        line = f"- id={c.id} | name={c.name} | value_type={c.value_type}"
        lines.append(line)

        if include_enum_for_select and c.value_type in ("SINGLE_SELECT", "MULTI_SELECT") and c.value_enum:
            enum_list = c.value_enum[:max_enum_items]
            # Keep enums on one line to reduce tokens; JSON-stringify for unambiguous quoting
            enums_str = json.dumps(enum_list, ensure_ascii=False)
            if c.value_enum and len(c.value_enum) > max_enum_items:
                enums_str = enums_str[:-1] + ', "..."]'  # indicate truncation
            lines.append(f"  allowed_values={enums_str}")

    return "\n".join(lines)

def format_exemplar(ex: RetrievalItem, *, include_full_transcript: bool = True) -> str:
    """
    Format a retrieved exemplar as a mini training pair:
      Transcript:
      <text>
      Output:
      <canonical json>
    """
    transcript = ex.transcript.strip() if include_full_transcript else ex.transcript.strip()
    out_json = idvalues_to_canonical_json(ex.gold_idvalues, indent=2)
    return (
        "Example (for format and normalization; do NOT copy values):\n"
        "Transcript:\n"
        f"{transcript}\n"
        "Output:\n"
        f"{out_json}\n"
    )


# -----------------------------
# Main builder
# -----------------------------

def build_messages_for_llama4(
    *,
    query_transcript: str,
    retrieved_examples: Sequence[RetrievalItem],
    candidate_result: CandidateResult,
    max_exemplars: int = 3,
    include_candidate_table: bool = True,
    include_exemplar_outputs: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build chat messages for Llama 4 Instruct.

    Returns messages suitable for:
      processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    Structure:
    - system: SYSTEM_MSG
    - user: candidate table + exemplars + query
    """
    # 1) System
    messages: List[Dict[str, Any]] = [{"role": "system", "content": [_txt(SYSTEM_MSG)]}]

    # 2) User content block
    parts: List[str] = []

    if include_candidate_table:
        parts.append(format_candidate_table(candidate_result.candidates))

    # Exemplars: include top-k retrieved, with gold outputs
    if include_exemplar_outputs and retrieved_examples:
        k = min(max_exemplars, len(retrieved_examples))
        parts.append("\nRetrieved Examples:")
        for i in range(k):
            parts.append(format_exemplar(retrieved_examples[i]))

    # Query
    parts.append("Now extract observations for the following transcript.")
    parts.append("Transcript:")
    parts.append(query_transcript.strip())
    parts.append("\nReturn ONLY the JSON list of {id,value} objects, nothing else.")

    user_text = "\n\n".join(parts)
    messages.append({"role": "user", "content": [_txt(user_text)]})

    return messages


# -----------------------------
# Optional: smaller variant (two-turn few-shot)
# -----------------------------

def build_messages_two_turn_fewshot(
    *,
    query_transcript: str,
    retrieved_examples: Sequence[RetrievalItem],
    candidate_result: CandidateResult,
    max_exemplars: int = 2,
) -> List[Dict[str, Any]]:
    """
    Alternative prompt structure:
    - system
    - (user, assistant) for each exemplar (shows output as assistant)
    - final user with candidate table + query transcript

    This sometimes improves format adherence in some chat models.
    """
    messages: List[Dict[str, Any]] = [{"role": "system", "content": [_txt(SYSTEM_MSG)]}]

    k = min(max_exemplars, len(retrieved_examples))
    for i in range(k):
        ex = retrieved_examples[i]
        messages.append({"role": "user", "content": [_txt("Transcript:\n" + ex.transcript.strip())]})
        messages.append({"role": "assistant", "content": [_txt(idvalues_to_canonical_json(ex.gold_idvalues, indent=2))]})

    user_parts: List[str] = []
    user_parts.append(format_candidate_table(candidate_result.candidates))
    user_parts.append("Transcript:\n" + query_transcript.strip())
    user_parts.append("Return ONLY valid JSON list of {id,value}.")
    messages.append({"role": "user", "content": [_txt("\n\n".join(user_parts))]})

    return messages
