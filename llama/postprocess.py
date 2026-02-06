"""
Post processing step to ensure valid JSON output
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

from define import observations_to_eval_dicts
from define import IDValue, Observation, SchemaIndex, parse_idvalue_json
from gatekeeper import (
    ValidationResult,
    validate_and_normalize_idvalues,
    expand_cleaned_to_observations,
)


# -----------------------------
# Deterministic repair helpers
# -----------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")  # remove trailing comma before } or ]
_SINGLE_QUOTES_RE = re.compile(r"(?<!\\)'")      # naive: replace single quotes with double quotes


def strip_code_fences(s: str) -> str:
    s2 = s.strip()
    if s2.startswith("```"):
        s2 = re.sub(r"^```[a-zA-Z]*\s*\n", "", s2)
        s2 = re.sub(r"\n```$", "", s2)
    return s2.strip()


def extract_first_json_array_or_object(s: str) -> str:
    s = s.strip()
    start_idx = None
    for i, ch in enumerate(s):
        if ch == "[" or ch == "{":
            start_idx = i
            break
    if start_idx is None:
        return s

    stack = []
    for j in range(start_idx, len(s)):
        ch = s[j]
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                break
            top = stack.pop()
            if (top == "[" and ch != "]") or (top == "{" and ch != "}"):
                break
            if not stack:
                return s[start_idx : j + 1]
    return s[start_idx:]


def normalize_common_json_issues(s: str) -> str:
    s2 = s.strip()
    s2 = strip_code_fences(s2)
    s2 = extract_first_json_array_or_object(s2)
    s2 = _TRAILING_COMMA_RE.sub(r"\1", s2)
    return s2.strip()


# -----------------------------
# Parse with deterministic repairs
# -----------------------------

@dataclass(frozen=True)
class ParseAttempt:
    ok: bool
    parsed: Optional[List[IDValue]]
    text_used: str
    error: Optional[str]


def parse_with_repairs(raw_text: str, max_attempts: int = 3) -> ParseAttempt:
    text = raw_text.strip()

    attempts: List[str] = [text]
    attempts.append(normalize_common_json_issues(text))

    if "'" in text and ('"' not in text or text.count("'") > text.count('"')):
        attempts.append(_SINGLE_QUOTES_RE.sub('"', attempts[-1]))

    uniq_attempts: List[str] = []
    seen = set()
    for a in attempts:
        if a not in seen and a.strip():
            uniq_attempts.append(a)
            seen.add(a)

    last_err: Optional[str] = None
    last_text: str = text

    for a in uniq_attempts[:max_attempts]:
        last_text = a
        try:
            parsed = parse_idvalue_json(a)
            return ParseAttempt(ok=True, parsed=parsed, text_used=a, error=None)
        except Exception as e:
            last_err = str(e)

    return ParseAttempt(ok=False, parsed=None, text_used=last_text, error=last_err)


# -----------------------------
# Optional LLM-based repair hook
# -----------------------------

class LLMRepairer:
    def repair_to_json(self, raw_text: str) -> str:
        raise NotImplementedError


# -----------------------------
# Unit-pair completion (NEW)
# -----------------------------

_UNIT_MARKERS = (" unit", " units")


def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _looks_like_concept_obj(x) -> bool:
    """
    Heuristic: concept objects usually have at least 'name' and 'value_type' attributes.
    """
    return hasattr(x, "name") and hasattr(x, "value_type")


def _iter_schema_concepts(schema: SchemaIndex):
    """
    Yield (concept_id, concept_obj) pairs from SchemaIndex without assuming internal field names.

    Strategy:
    1) Try common attributes first (concepts/id_to_concept/by_id/targets).
    2) If not found, introspect schema.__dict__ to find a dict that looks like {id -> concept}.
    3) As a last resort, try iterating schema (if it yields ids) and calling schema.get(id).
    """
    # 1) Common dict-like attributes
    for attr in ("concepts", "id_to_concept", "by_id", "targets"):
        m = getattr(schema, attr, None)
        if isinstance(m, dict) and m:
            # verify it looks like a concept dict
            sample_val = next(iter(m.values()))
            if _looks_like_concept_obj(sample_val):
                for cid, c in m.items():
                    yield str(cid), c
                return

    # 2) Reflection: search schema.__dict__ for a dict that looks like id->concept
    d = getattr(schema, "__dict__", {}) or {}
    for k, v in d.items():
        if not isinstance(v, dict) or not v:
            continue

        # check a few samples
        vals = list(v.values())
        sample = vals[0]
        if not _looks_like_concept_obj(sample):
            continue

        # also check keys look like ids (mostly digits or strings)
        keys = list(v.keys())
        key0 = keys[0]
        if not (isinstance(key0, (str, int))):
            continue

        # This is very likely the concept table
        for cid, c in v.items():
            yield str(cid), c
        return

    # 3) Last resort: try iterating schema and calling schema.get(id)
    try:
        for cid in schema:
            c = schema.get(str(cid))
            if c is not None:
                yield str(cid), c
        return
    except Exception:
        pass

    # If we get here, SchemaIndex is not enumerable with current info.
    raise AttributeError(
        "Could not iterate SchemaIndex concepts. "
        "SchemaIndex hides concept storage; please expose an iterable of ids or an id->concept dict, "
        "or add a method like schema.items() returning (id, concept) pairs."
    )



def build_unit_pairs_from_schema(schema: SchemaIndex) -> Dict[str, str]:
    """
    Build measure_id -> unit_id pairs from schema concept names.
    Uses _iter_schema_concepts(schema) so it doesn't assume schema.concepts exists.
    """
    non_unit_by_name: Dict[str, List[str]] = {}
    unit_concepts: List[Tuple[str, str]] = []

    for cid, c in _iter_schema_concepts(schema):
        name_norm = _norm_name(getattr(c, "name", ""))
        is_unit = any(m in name_norm for m in _UNIT_MARKERS) or name_norm.endswith(" unit")
        if is_unit:
            unit_concepts.append((cid, name_norm))
        else:
            non_unit_by_name.setdefault(name_norm, []).append(cid)

    pairs: Dict[str, str] = {}

    for unit_id, unit_name_norm in unit_concepts:
        base = unit_name_norm
        if base.endswith(" units"):
            base = base[: -len(" units")].strip()
        elif base.endswith(" unit"):
            base = base[: -len(" unit")].strip()
        else:
            continue

        measure_ids = non_unit_by_name.get(base)
        if not measure_ids:
            continue

        measure_id = sorted(measure_ids, key=lambda x: int(x) if str(x).isdigit() else str(x))[0]
        pairs[measure_id] = unit_id

    return pairs



def _enum_to_pattern(ev: str) -> Optional[re.Pattern]:
    if not isinstance(ev, str):
        return None
    tok = ev.strip()
    if not tok or len(tok) > 40:
        return None
    return re.compile(re.escape(tok), re.IGNORECASE)


def detect_unit_value_from_transcript(transcript: str, unit_enum_values: Sequence[Any]) -> Optional[str]:
    """
    Find an enum token in transcript. Prefer longer tokens first.
    Return the exact enum token (so scoring matches reference).
    """
    t = transcript or ""
    enums = [e for e in unit_enum_values if isinstance(e, str) and e.strip()]
    enums.sort(key=len, reverse=True)

    for ev in enums:
        pat = _enum_to_pattern(ev)
        if pat and pat.search(t):
            return ev

    # small aliases (only if schema includes the canonical enum)
    t_lower = t.lower()
    alias_groups = [
        ("ml", {"ml", "mL", "milliliter", "milliliters", "millilitre", "millilitres"}),
        ("cc", {"cc", "c.c.", "c c"}),
    ]
    for canon, variants in alias_groups:
        if any(v.lower() in t_lower for v in variants):
            for ev in enums:
                if ev.strip().lower() == canon:
                    return ev

    return None


def add_all_missing_units(
    *,
    transcript: str,
    preds: List[IDValue],
    schema: SchemaIndex,
    candidate_ids: Optional[Set[str]] = None,
    unit_pairs: Optional[Dict[str, str]] = None,
) -> Tuple[List[IDValue], Optional[Set[str]]]:
    """
    For each (measure_id -> unit_id):
      if measure predicted and unit missing and unit token found in transcript -> add unit prediction.

    Also adds unit_id to candidate_ids if provided (important if you restrict by candidates).
    """
    if unit_pairs is None:
        unit_pairs = build_unit_pairs_from_schema(schema)

    have = {iv.id for iv in preds}
    new_preds = list(preds)

    for measure_id, unit_id in unit_pairs.items():
        if measure_id not in have:
            continue
        if unit_id in have:
            continue

        unit_concept = schema.get(unit_id)
        unit_val = detect_unit_value_from_transcript(transcript, unit_concept.value_enum or [])
        if unit_val is None:
            continue

        new_preds.append(IDValue(id=unit_id, value=unit_val))
        have.add(unit_id)

        if candidate_ids is not None:
            candidate_ids = set(candidate_ids)
            candidate_ids.add(unit_id)

    return new_preds, candidate_ids


# -----------------------------
# End-to-end postprocess outputs
# -----------------------------

@dataclass(frozen=True)
class PostprocessOutput:
    parse_ok: bool
    parse_error: Optional[str]
    raw_text: str
    repaired_text_used: str
    parsed_idvalues: List[IDValue]
    validation: ValidationResult
    expanded_observations: List[Observation]


def postprocess_prediction(
    raw_text: str,
    schema: SchemaIndex,
    *,
    candidate_ids: Optional[set[str]] = None,
    drop_none_values: bool = True,
    repairer: Optional[LLMRepairer] = None,
) -> PostprocessOutput:
    """
    Original behavior (kept).
    """
    attempt = parse_with_repairs(raw_text)

    parsed: List[IDValue] = []
    repaired_text = attempt.text_used

    if not attempt.ok and repairer is not None:
        repaired_text = repairer.repair_to_json(raw_text)
        attempt2 = parse_with_repairs(repaired_text)
        if attempt2.ok and attempt2.parsed is not None:
            attempt = attempt2

    if attempt.ok and attempt.parsed is not None:
        parsed = attempt.parsed

    validation = validate_and_normalize_idvalues(
        preds=parsed,
        schema=schema,
        candidate_ids=candidate_ids,
        drop_none_values=drop_none_values,
    )
    expanded = expand_cleaned_to_observations(validation.cleaned, schema) if validation.cleaned else []

    return PostprocessOutput(
        parse_ok=attempt.ok,
        parse_error=attempt.error if not attempt.ok else None,
        raw_text=raw_text,
        repaired_text_used=repaired_text,
        parsed_idvalues=parsed,
        validation=validation,
        expanded_observations=expanded,
    )


def postprocess_prediction_with_units(
    raw_text: str,
    transcript: str,
    schema: SchemaIndex,
    *,
    candidate_ids: Optional[set[str]] = None,
    drop_none_values: bool = True,
    repairer: Optional[LLMRepairer] = None,
    unit_pairs: Optional[Dict[str, str]] = None,
) -> PostprocessOutput:
    """
    NEW behavior:
      parse/repair -> add paired unit concepts (ALL) -> validate/normalize -> expand
    """
    attempt = parse_with_repairs(raw_text)

    parsed: List[IDValue] = []
    repaired_text = attempt.text_used

    if not attempt.ok and repairer is not None:
        repaired_text = repairer.repair_to_json(raw_text)
        attempt2 = parse_with_repairs(repaired_text)
        if attempt2.ok and attempt2.parsed is not None:
            attempt = attempt2

    if attempt.ok and attempt.parsed is not None:
        parsed = attempt.parsed

    parsed2, cand2 = add_all_missing_units(
        transcript=transcript,
        preds=parsed,
        schema=schema,
        candidate_ids=candidate_ids,
        unit_pairs=unit_pairs,
    )

    validation = validate_and_normalize_idvalues(
        preds=parsed2,
        schema=schema,
        candidate_ids=cand2,
        drop_none_values=drop_none_values,
    )
    expanded = expand_cleaned_to_observations(validation.cleaned, schema) if validation.cleaned else []

    return PostprocessOutput(
        parse_ok=attempt.ok,
        parse_error=attempt.error if not attempt.ok else None,
        raw_text=raw_text,
        repaired_text_used=repaired_text,
        parsed_idvalues=parsed2,
        validation=validation,
        expanded_observations=expanded,
    )


def to_eval_jsonl_record(
    example_id: str,
    post: "PostprocessOutput",
    include_name: bool = False,
) -> Dict[str, Any]:
    return {
        "id": str(example_id),
        "observations": observations_to_eval_dicts(post.expanded_observations, include_name=include_name),
    }
