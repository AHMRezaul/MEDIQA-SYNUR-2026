"""
Step 2: Preprocess the training knowledge base (KB) from train.jsonl

What this does:
- Reads /data/train.jsonl (JSON Lines)
- Parses the double-encoded "observations" field (a JSON string inside the JSONL record)
- Produces a list of KBExample objects with:
    - id
    - transcript  (the retrievable text)
    - gold_idvalues  (canonical supervision target: [{"id":..., "value":...}, ...])
    - gold_observations (training-style objects as in data)
    - concept_signature (set of concept ids)
    - concept_text (optional: names of concepts, useful for concept-aware retrieval later)
- Produces basic dataset stats for sanity checks (counts, unique ids)

Notes:
- This is preprocessing only. No embedding / retrieval index building yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from define import IDValue, Observation, SchemaIndex


# -----------------------------
# KB Example Definition
# -----------------------------

@dataclass(frozen=True)
class KBExample:
    example_id: str
    transcript: str
    gold_idvalues: List[IDValue]
    gold_observations: List[Observation]
    concept_signature: Set[str]
    concept_text: str  # concept names joined (optional, for retrieval)


@dataclass(frozen=True)
class KBStats:
    num_examples: int
    num_total_observations: int
    num_unique_concept_ids: int
    unique_concept_ids: Set[str]
    bad_lines: int
    bad_observation_payloads: int


# -----------------------------
# Parsing helpers
# -----------------------------

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def parse_observations_field(obs_field: Any) -> List[Dict[str, Any]]:
    """
    The dataset stores "observations" as a JSON string (most of the time).
    This function returns it as a list[dict].

    Accepts:
    - obs_field as string containing JSON array
    - obs_field as already-parsed list (just in case)
    """
    if obs_field is None:
        return []

    # Common case: string of JSON
    if isinstance(obs_field, str):
        obs_field = obs_field.strip()
        if not obs_field:
            return []
        parsed = json.loads(obs_field)
        if not isinstance(parsed, list):
            raise ValueError("observations JSON did not parse to a list")
        return parsed

    # Sometimes you may already have list (depending on upstream)
    if isinstance(obs_field, list):
        return obs_field

    raise ValueError(f"Unsupported observations field type: {type(obs_field)}")


def obs_dicts_to_targets(
    obs_list: List[Dict[str, Any]],
) -> Tuple[List[Observation], List[IDValue]]:
    """
    Convert raw observation dicts from train.jsonl into:
    - training-style Observation objects
    - canonical IDValue objects (recommended prediction target)

    This does not validate against schema yet (that happens later).
    """
    observations: List[Observation] = []
    idvalues: List[IDValue] = []

    for i, o in enumerate(obs_list):
        if not isinstance(o, dict):
            # Skip malformed entry
            continue

        cid = _safe_str(o.get("id", "")).strip()
        name = _safe_str(o.get("name", "")).strip()
        vtype = _safe_str(o.get("value_type", "")).strip()

        # "value" is required in practice, but allow missing -> None
        value = o.get("value", None)

        if not cid:
            # Skip if no id
            continue

        # Keep as-is; schema-based corrections come later
        observations.append(Observation(id=cid, name=name, value_type=vtype, value=value))
        idvalues.append(IDValue(id=cid, value=value))

    return observations, idvalues


# -----------------------------
# KB Builder
# -----------------------------

def build_kb_from_train_jsonl(
    train_jsonl_path: str,
    schema: Optional[SchemaIndex] = None,
    use_schema_names_for_concept_text: bool = True,
) -> Tuple[List[KBExample], KBStats]:
    """
    Build a KB from train.jsonl.

    If schema is provided:
    - concept_text will be composed using schema names for each id
    - you can also sanity-check name/value_type mismatches later

    If schema is None:
    - concept_text will use whatever 'name' appears in training file
    """
    kb: List[KBExample] = []

    unique_ids: Set[str] = set()
    total_obs = 0
    bad_lines = 0
    bad_obs_payloads = 0

    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            if not isinstance(record, dict):
                bad_lines += 1
                continue

            ex_id = _safe_str(record.get("id", "")).strip()
            transcript = _safe_str(record.get("transcript", "")).strip()

            # Parse observations (double-encoded JSON string)
            try:
                obs_list = parse_observations_field(record.get("observations", None))
            except Exception:
                bad_obs_payloads += 1
                obs_list = []

            gold_obs, gold_idvalues = obs_dicts_to_targets(obs_list)

            concept_sig: Set[str] = set()
            concept_names: List[str] = []

            for o in gold_obs:
                concept_sig.add(o.id)
                unique_ids.add(o.id)

                if schema is not None and use_schema_names_for_concept_text and schema.has_id(o.id):
                    concept_names.append(schema.get(o.id).name)
                else:
                    # fall back to training-provided name if schema is absent
                    if o.name:
                        concept_names.append(o.name)

            total_obs += len(gold_obs)

            # concept_text used later for concept-aware retrieval (optional)
            concept_text = " | ".join(sorted(set(concept_names)))

            kb.append(
                KBExample(
                    example_id=ex_id or f"line_{line_no}",
                    transcript=transcript,
                    gold_idvalues=gold_idvalues,
                    gold_observations=gold_obs,
                    concept_signature=concept_sig,
                    concept_text=concept_text,
                )
            )

    stats = KBStats(
        num_examples=len(kb),
        num_total_observations=total_obs,
        num_unique_concept_ids=len(unique_ids),
        unique_concept_ids=unique_ids,
        bad_lines=bad_lines,
        bad_observation_payloads=bad_obs_payloads,
    )

    return kb, stats


# -----------------------------
# Optional: Sanity checks vs schema
# -----------------------------

@dataclass(frozen=True)
class SchemaMismatch:
    concept_id: str
    train_name: str
    schema_name: str
    train_value_type: str
    schema_value_type: str

def find_schema_mismatches(
    kb: List[KBExample],
    schema: SchemaIndex,
    max_report: int = 50,
) -> List[SchemaMismatch]:
    """
    Find cases where training-provided name/value_type differ from schema.
    Useful to learn whether you should trust schema as canonical (usually yes).
    """
    mismatches: List[SchemaMismatch] = []

    for ex in kb:
        for o in ex.gold_observations:
            if not schema.has_id(o.id):
                continue
            c = schema.get(o.id)

            # Compare only if training has non-empty fields
            train_name = (o.name or "").strip()
            train_vtype = (o.value_type or "").strip()

            name_diff = bool(train_name) and (train_name != c.name)
            vtype_diff = bool(train_vtype) and (train_vtype != c.value_type)

            if name_diff or vtype_diff:
                mismatches.append(
                    SchemaMismatch(
                        concept_id=o.id,
                        train_name=train_name,
                        schema_name=c.name,
                        train_value_type=train_vtype,
                        schema_value_type=c.value_type,
                    )
                )
                if len(mismatches) >= max_report:
                    return mismatches

    return mismatches

