"""Offline model-experiment scoring; never imported by the training runner.

Local alignment and protected-text checks are mechanical evidence, not semantic
judgments. Ambiguous alignments remain unresolved for a separate blinded review.
"""
from collections import Counter
from difflib import SequenceMatcher


def _span_output(source, output, start, end):
    """Map a source span only when both boundaries are unambiguous."""
    positions = {start: set(), end: set()}
    for tag, i, j, a, b in SequenceMatcher(None, source, output, autojunk=False).get_opcodes():
        for boundary in positions:
            if tag == "equal" and i <= boundary <= j:
                positions[boundary].add(a + boundary - i)
            elif tag != "equal":
                if i == boundary:
                    positions[boundary].add(a)
                if j == boundary:
                    positions[boundary].add(b)
    if any(len(value) != 1 for value in positions.values()):
        return None
    left, right = next(iter(positions[start])), next(iter(positions[end]))
    return output[left:right] if left <= right else None


def validate_annotations(cases):
    seen = set()
    for case in cases:
        if not isinstance(case.get("id"), str) or case["id"] in seen:
            raise ValueError("Missing or duplicate case id")
        seen.add(case["id"])
        text = case["text"]
        refs = case["acceptable_outputs"]
        if not isinstance(refs, list) or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
            raise ValueError("Invalid reference list")
        cursor, pieces = 0, []
        for span in case.get("targets", []):
            start, end = span["start"], span["end"]
            if not (type(start) is int and type(end) is int and cursor <= start < end <= len(text)):
                raise ValueError("Invalid or overlapping target offsets")
            if text[start:end] != span["source"]:
                raise ValueError("Target offsets differ from source")
            replacements = span["acceptable_outputs"]
            if not isinstance(replacements, list) or not replacements or any(not isinstance(x, str) for x in replacements):
                raise ValueError("Invalid target replacements")
            pieces.extend((text[cursor:start], replacements[0]))
            cursor = end
        pieces.append(text[cursor:])
        if refs and "".join(pieces) != refs[0]:
            raise ValueError("Target annotations do not replay to the primary reference")
        for span in case.get("protected_spans", []):
            start, end = span["start"], span["end"]
            if not (type(start) is int and type(end) is int and 0 <= start < end <= len(text)) or text[start:end] != span["text"]:
                raise ValueError("Invalid protected source span")


def evaluate(cases, predictions, scorer, surface):
    validate_annotations(cases)
    ids = {case["id"] for case in cases}
    by_id = {}
    for row in predictions:
        if row.get("id") not in ids or row["id"] in by_id:
            raise ValueError("Unknown or duplicate prediction")
        by_id[row["id"]] = row
    scored_cases = [case for case in cases if case["acceptable_outputs"]]
    result = scorer(scored_cases, [row for row in predictions if row["id"] in {c["id"] for c in scored_cases}])
    states, local, protections, extra = Counter(), Counter(), Counter(), []
    for case in cases:
        row = by_id.get(case["id"], {"status": "missing", "prediction": ""})
        output = row.get("prediction", "")
        valid = row.get("status") == "ok" and isinstance(output, str) and bool(output.strip())
        states[row.get("status", "missing")] += 1
        if not isinstance(output, str):
            output = ""
        targets = []
        for span in case.get("targets", []):
            observed = _span_output(case["text"], output, span["start"], span["end"]) if valid else None
            if not valid:
                state = "failed"
            elif observed is None:
                state = "unresolved_alignment"
            elif surface(observed) in {surface(x) for x in span["acceptable_outputs"]}:
                state = "matched"
            else:
                state = "unmatched"
            local[state] += 1
            targets.append(dict(span, observed=observed, alignment_status=state))
        protected = []
        for span in case.get("protected_spans", []):
            observed = _span_output(case["text"], output, span["start"], span["end"]) if valid else None
            state = "failed" if not valid else "unresolved_alignment" if observed is None else "preserved" if observed == span["text"] else "changed"
            protections[state] += 1
            protected.append(dict(span, observed=observed, alignment_status=state))
        changes_outside = []
        for tag, left, right, out_left, out_right in SequenceMatcher(None, case["text"], output, autojunk=False).get_opcodes():
            if tag != "equal" and not any(span["start"] <= left and right <= span["end"] for span in case.get("targets", [])):
                changes_outside.append({"operation": tag, "source_start": left, "source_end": right,
                                        "source": case["text"][left:right], "output": output[out_left:out_right]})
        extra.append({"id": case["id"], "targets": targets, "protected_spans": protected,
                      "edits_outside_targets": changes_outside, "exact_source_preserved": valid and output == case["text"]})
    target_total = sum(local.values())
    result.update(total_cases=len(cases), reference_coverage={"scored": len(scored_cases), "pending": len(cases)-len(scored_cases)},
                  execution_statuses=dict(states), local_targets={"total": target_total, **local,
                  "mechanically_matched_rate": local["matched"]/target_total if target_total else None},
                  protected_spans=dict(protections), mechanical_details=extra,
                  semantic_review="Separate blinded review required; edit/alignment checks do not establish semantic correctness.")
    return result


def paired_changes(base, adapted):
    left = {row["id"]: row for row in base["cases"]}
    right = {row["id"]: row for row in adapted["cases"]}
    if set(left) != set(right):
        raise ValueError("Paired systems must cover the same scored cases")
    return {"new_matches": [key for key in left if not left[key]["surface_correct"] and right[key]["surface_correct"]],
            "regressions": [key for key in left if left[key]["surface_correct"] and not right[key]["surface_correct"]],
            "both_match": sum(left[key]["surface_correct"] and right[key]["surface_correct"] for key in left),
            "both_unmatched": sum(not left[key]["surface_correct"] and not right[key]["surface_correct"] for key in left)}
