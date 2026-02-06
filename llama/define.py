"""
Step 1: Canonical target definition (recommended)

Goal:
- During LLM inference, we will ask for ONLY: [{"id": <concept_id>, "value": <value>}, ...]
- Then we will deterministically expand to the training-style objects using synur_schema.json:
  [{"id": ..., "name": ..., "value_type": ..., "value": ...}, ...]

This file defines:
- The canonical prediction target (IDValue)
- The expanded output target (Observation)
- Validators for the *shape* of the canonical output (not semantic validation yet)
- Deterministic expansion using the schema
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import asdict
from typing import Any, Dict, List, Optional


# -----------------------------
# Canonical & Expanded Targets
# -----------------------------

Value = Union[str, int, float, List[str], List[int], List[float], None]


@dataclass(frozen=True)
class IDValue:
    """Canonical prediction target (what the LLM should output)."""
    id: str
    value: Value


@dataclass(frozen=True)
class Observation:
    """Expanded target (training-style object)."""
    id: str
    name: str
    value_type: str
    value: Value


# -----------------------------
# Schema Loading & Indexing
# -----------------------------

@dataclass(frozen=True)
class ConceptDef:
    id: str
    name: str
    value_type: str
    value_enum: Optional[List[str]] = None


class SchemaIndex:
    """
    Loads synur_schema.json and builds lookup tables:
      - id -> (name, value_type, value_enum)
    """

    def __init__(self, concepts: List[ConceptDef]) -> None:
        self._by_id: Dict[str, ConceptDef] = {c.id: c for c in concepts}

    @classmethod
    def from_json_file(cls, schema_path: str) -> "SchemaIndex":
        with open(schema_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Accept either:
        # - {"concepts": [ ... ]} or
        # - [ ... ] (list root)
        if isinstance(raw, dict) and "concepts" in raw:
            raw_concepts = raw["concepts"]
        elif isinstance(raw, list):
            raw_concepts = raw
        else:
            raise ValueError("Unexpected schema format. Expected list or dict with key 'concepts'.")

        concepts: List[ConceptDef] = []
        for item in raw_concepts:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id", "")).strip()
            name = str(item.get("name", "")).strip()
            vtype = str(item.get("value_type", "")).strip()
            venum = item.get("value_enum", None)
            if venum is not None and not isinstance(venum, list):
                venum = None
            venum = [str(x) for x in venum] if isinstance(venum, list) else None

            if not cid or not name or not vtype:
                # Skip malformed entries
                continue

            concepts.append(ConceptDef(id=cid, name=name, value_type=vtype, value_enum=venum))

        if not concepts:
            raise ValueError("No concepts loaded from schema. Check file content/format.")

        return cls(concepts)

    def has_id(self, concept_id: str) -> bool:
        return concept_id in self._by_id

    def get(self, concept_id: str) -> ConceptDef:
        return self._by_id[concept_id]

    def all_ids(self) -> List[str]:
        return list(self._by_id.keys())


# -----------------------------
# Canonical Output Parsing
# -----------------------------

def parse_idvalue_json(text: str) -> List[IDValue]:
    """
    Parse canonical LLM output from a JSON string.

    Expected JSON shape:
      [
        {"id": "38", "value": "Yes"},
        {"id": "33", "value": 98},
        ...
      ]

    This does NOT do semantic validation against schema yet.
    It only validates that output is JSON list[dict] with keys id/value.
    """
    data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError("Canonical output must be a JSON list.")

    out: List[IDValue] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be an object/dict.")
        if "id" not in item:
            raise ValueError(f"Item {i} missing required key 'id'.")
        # "value" key is required (can be None)
        if "value" not in item:
            raise ValueError(f"Item {i} missing required key 'value'.")

        cid = str(item["id"]).strip()
        if not cid:
            raise ValueError(f"Item {i} has empty 'id'.")

        out.append(IDValue(id=cid, value=item["value"]))

    return out


# -----------------------------
# Deterministic Expansion
# -----------------------------

def expand_idvalues(
    preds: List[IDValue],
    schema: SchemaIndex,
    drop_unknown_ids: bool = True,
) -> Tuple[List[Observation], List[IDValue]]:
    """
    Expand canonical predictions to training-style observations using schema.

    Returns:
      - expanded observations (known ids only if drop_unknown_ids=True)
      - unknown predictions (ids not found in schema)

    Note: Semantic/type validation happens in later steps.
    """
    expanded: List[Observation] = []
    unknown: List[IDValue] = []

    for p in preds:
        if not schema.has_id(p.id):
            if drop_unknown_ids:
                unknown.append(p)
                continue
            else:
                # If you ever want to keep unknown ids, you can set placeholder fields.
                unknown.append(p)
                expanded.append(Observation(id=p.id, name="", value_type="", value=p.value))
                continue

        c = schema.get(p.id)
        expanded.append(Observation(id=c.id, name=c.name, value_type=c.value_type, value=p.value))

    return expanded, unknown


# -----------------------------
# Optional: Canonical Serialization
# -----------------------------

def idvalues_to_json(preds: List[IDValue]) -> str:
    """Serialize canonical predictions to a JSON string (pretty-printed)."""
    payload = [{"id": p.id, "value": p.value} for p in preds]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def observations_to_json(obs: List[Observation]) -> str:
    """Serialize expanded observations to a JSON string (pretty-printed)."""
    payload = [{"id": o.id, "name": o.name, "value_type": o.value_type, "value": o.value} for o in obs]
    return json.dumps(payload, ensure_ascii=False, indent=2)

def observations_to_eval_dicts(obs: List["Observation"], include_name: bool = False) -> List[Dict[str, Any]]:
    """
    Convert expanded Observation objects to dicts compatible with the evaluator.

    Evaluator requires: id, value_type, value
    name is optional (ignored by scorer).
    """
    out: List[Dict[str, Any]] = []
    for o in obs:
        d = {"id": o.id, "value_type": o.value_type, "value": o.value}
        if include_name:
            d["name"] = o.name
        out.append(d)
    return out
