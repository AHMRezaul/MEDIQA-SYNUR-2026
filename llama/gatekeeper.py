"""
Step 3: Prepare the schema as an ontology + constraint source

What this does:
- Loads synur_schema.json into SchemaIndex (from Step 1)
- Builds fast lookup maps (already in SchemaIndex) and additional utilities:
    - type/enums getters
    - value normalization helpers (especially for unit encoding)
    - per-type validation and normalization of predicted values
- Provides a single entry point:
    validate_and_normalize_idvalues(preds, schema, candidate_ids=None)
  which:
    - drops unknown ids
    - (optionally) drops ids not in candidate_ids
    - validates value shape based on schema.value_type
    - normalizes values to match schema enums and numeric conventions
    - deduplicates by id with a clear rule

Notes:
- This step focuses on *schema-level* validation (id exists, enum membership, numeric parsing).
- It does not do transcript-grounding (i.e., whether the value is actually supported by text).
  That’s a later concern.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from define import IDValue, Observation, SchemaIndex, ConceptDef


# -----------------------------
# Normalization Helpers
# -----------------------------

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def normalize_whitespace(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()

def normalize_unit_token(s: str) -> str:
    """
    Conservative normalization:
    - Fix common mojibake artifacts like 'Â°F' -> '°F' and 'Â°C' -> '°C'
    - Do NOT map to 'F'/'C' because the evaluator expects exact enum tokens from reference.
    """
    s = s.strip()
    s = s.replace("Â°C", "°C").replace("Â°F", "°F")
    s = s.replace("\\u00c2\\u00b0C", "°C").replace("\\u00c2\\u00b0F", "°F")
    return s

def try_parse_number(x: Any) -> Optional[Union[int, float]]:
    """
    Parse a numeric value. Accepts:
    - int/float directly
    - strings like "98", "98.6", "98 percent", "98%" -> 98 / 98.6
    Returns None if parsing fails.
    """
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # Extract the first numeric token from the string
        m = _NUM_RE.search(s)
        if not m:
            return None
        token = m.group(0)
        try:
            # Prefer int if it looks integral
            if "." in token:
                return float(token)
            return int(token)
        except Exception:
            try:
                return float(token)
            except Exception:
                return None
    return None


# -----------------------------
# Enum Matching Helpers
# -----------------------------

def build_enum_normalizer(enum_values: List[str]) -> Dict[str, str]:
    """
    Build a map from normalized forms -> canonical enum token.
    This allows tolerant matching: casefold + whitespace normalization + unit normalization.
    """
    norm_map: Dict[str, str] = {}
    for v in enum_values:
        canon = normalize_unit_token(normalize_whitespace(str(v)))
        key = canon.casefold()
        norm_map[key] = canon
    return norm_map

def match_enum(value: Any, enum_norm_map: Dict[str, str]) -> Optional[str]:
    """
    Try to match a predicted value to one of the enum options.
    Returns the canonical enum string if matched, else None.
    """
    if value is None:
        return None

    # If value is a number, enums are usually strings; try exact string match.
    s = normalize_unit_token(normalize_whitespace(str(value)))
    key = s.casefold()
    return enum_norm_map.get(key)


# -----------------------------
# Validation Result Types
# -----------------------------

@dataclass(frozen=True)
class ValidationIssue:
    concept_id: str
    reason: str
    raw_value: Any


@dataclass(frozen=True)
class ValidationResult:
    cleaned: List[IDValue]
    dropped: List[ValidationIssue]


# -----------------------------
# Per-type validation/normalization
# -----------------------------

def validate_single_select(value: Any, enum_values: Optional[List[str]]) -> Optional[str]:
    if enum_values is None:
        # No enum provided; treat as a free string but normalize whitespace
        if value is None:
            return None
        return normalize_unit_token(normalize_whitespace(str(value)))

    enum_norm = build_enum_normalizer(enum_values)
    matched = match_enum(value, enum_norm)
    return matched

def validate_multi_select(value: Any, enum_values: Optional[List[str]]) -> Optional[List[str]]:
    # Must be a list (or a single string we can promote to a singleton list)
    if value is None:
        return None

    items: List[Any]
    if isinstance(value, list):
        items = value
    else:
        # Promote scalar to list (common LLM mistake)
        items = [value]

    # Normalize & filter empties
    cleaned_items: List[str] = []
    if enum_values is None:
        for it in items:
            s = normalize_unit_token(normalize_whitespace(str(it)))
            if s:
                cleaned_items.append(s)
        return cleaned_items if cleaned_items else None

    enum_norm = build_enum_normalizer(enum_values)
    for it in items:
        matched = match_enum(it, enum_norm)
        if matched is not None:
            cleaned_items.append(matched)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    deduped: List[str] = []
    for s in cleaned_items:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    return deduped if deduped else None

def validate_numeric(value: Any) -> Optional[Union[int, float]]:
    return try_parse_number(value)

def validate_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    return normalize_unit_token(normalize_whitespace(str(value)))


# -----------------------------
# Main Schema-driven Validator
# -----------------------------

def validate_and_normalize_idvalues(
    preds: List[IDValue],
    schema: SchemaIndex,
    candidate_ids: Optional[Set[str]] = None,
    drop_none_values: bool = True,
) -> ValidationResult:
    """
    Validate and normalize a list of (id, value) predictions against schema.

    Rules:
    - Drop ids not in schema
    - If candidate_ids provided, drop ids not in candidate set
    - Normalize values according to value_type and value_enum
    - Optionally drop items whose normalized value becomes None
    - Deduplicate by id:
        keep the first valid occurrence; later duplicates are dropped
        (You can change policy later if needed.)

    Returns cleaned predictions + a list of dropped issues.
    """
    cleaned: List[IDValue] = []
    dropped: List[ValidationIssue] = []

    seen_ids: Set[str] = set()

    for p in preds:
        cid = (p.id or "").strip()
        if not cid:
            dropped.append(ValidationIssue(concept_id="", reason="empty_id", raw_value=p.value))
            continue

        if not schema.has_id(cid):
            dropped.append(ValidationIssue(concept_id=cid, reason="unknown_id", raw_value=p.value))
            continue

        if candidate_ids is not None and cid not in candidate_ids:
            dropped.append(ValidationIssue(concept_id=cid, reason="not_in_candidate_set", raw_value=p.value))
            continue

        if cid in seen_ids:
            dropped.append(ValidationIssue(concept_id=cid, reason="duplicate_id", raw_value=p.value))
            continue

        concept = schema.get(cid)
        vtype = concept.value_type
        venum = concept.value_enum

        norm_value: Any = None

        if vtype == "SINGLE_SELECT":
            norm_value = validate_single_select(p.value, venum)
            if norm_value is None and venum is not None:
                dropped.append(ValidationIssue(concept_id=cid, reason="single_select_not_in_enum", raw_value=p.value))
                continue

        elif vtype == "MULTI_SELECT":
            norm_value = validate_multi_select(p.value, venum)
            if norm_value is None and venum is not None:
                dropped.append(ValidationIssue(concept_id=cid, reason="multi_select_no_valid_enum_items", raw_value=p.value))
                continue

        elif vtype == "NUMERIC":
            norm_value = validate_numeric(p.value)
            if norm_value is None:
                dropped.append(ValidationIssue(concept_id=cid, reason="numeric_parse_failed", raw_value=p.value))
                continue

        elif vtype == "STRING":
            norm_value = validate_string(p.value)
            if norm_value is None:
                dropped.append(ValidationIssue(concept_id=cid, reason="string_empty_or_none", raw_value=p.value))
                continue

        else:
            dropped.append(ValidationIssue(concept_id=cid, reason=f"unknown_value_type:{vtype}", raw_value=p.value))
            continue

        if drop_none_values and norm_value is None:
            dropped.append(ValidationIssue(concept_id=cid, reason="normalized_value_none", raw_value=p.value))
            continue

        cleaned.append(IDValue(id=cid, value=norm_value))
        seen_ids.add(cid)

    return ValidationResult(cleaned=cleaned, dropped=dropped)


# -----------------------------
# Expansion to full training-style objects (schema is canonical)
# -----------------------------

def expand_cleaned_to_observations(
    cleaned: List[IDValue],
    schema: SchemaIndex,
) -> List[Observation]:
    """
    Expand cleaned canonical id/values into full Observation objects.
    Uses schema as source of truth for name and value_type.
    """
    out: List[Observation] = []
    for p in cleaned:
        c = schema.get(p.id)
        out.append(Observation(id=c.id, name=c.name, value_type=c.value_type, value=p.value))
    return out


# -----------------------------
# Optional: Schema diagnostics
# -----------------------------

@dataclass(frozen=True)
class SchemaSummary:
    num_concepts: int
    value_type_counts: Dict[str, int]
    num_with_enum: int
    max_enum_size: int

def summarize_schema(schema: SchemaIndex) -> SchemaSummary:
    vtc: Dict[str, int] = {}
    with_enum = 0
    max_enum = 0
    for cid in schema.all_ids():
        c = schema.get(cid)
        vtc[c.value_type] = vtc.get(c.value_type, 0) + 1
        if c.value_enum:
            with_enum += 1
            max_enum = max(max_enum, len(c.value_enum))
    return SchemaSummary(
        num_concepts=len(schema.all_ids()),
        value_type_counts=vtc,
        num_with_enum=with_enum,
        max_enum_size=max_enum,
    )

