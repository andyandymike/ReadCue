"""Restrictive audit corrections over frozen exploratory policies.

No new dictionary entries or model execution. Frozen v4 remains reproducible;
this adapter withdraws unsupported inferences and carries generated readings
through the existing model-edit guard. Stage edits retain stage-input offsets.
"""
from __future__ import annotations

import re

from . import comment_policy_v2 as comment
from . import context_reading_policy_v3 as context
from . import reading_policy as reading
from . import semantic_emoji_policy as semantic


POLICY_VERSION = "audit-corrected-v1"
HISTORICAL_BASE = "semantic-context-v4"
_CONNECTOR = re.compile(r"(?:是|为)[ \t]*[:：=]?[ \t]*$")
_NUMERAL = re.compile(r"[0-9零〇一二两三四五六七八九十百千万亿]$")


def _rebuild(source, edits):
    cursor, parts, generated, shift = 0, [], [], 0
    for edit in edits:
        start, end = edit["start"], edit["end"]
        if not cursor <= start < end <= len(source) or source[start:end] != edit["original"]:
            raise ValueError("invalid stage edit")
        parts.extend((source[cursor:start], edit["replacement"]))
        if edit["replacement"]:
            generated.append((start + shift, start + shift + len(edit["replacement"])))
        shift += len(edit["replacement"]) - (end - start)
        cursor = end
    return "".join(parts) + source[cursor:], generated


def _reverse(source, result):
    inverse, shift = [], 0
    for edit in result["edits"]:
        left = edit["start"] + shift
        inverse.append(dict(start=left, end=left + len(edit["replacement"]),
                            original=edit["replacement"], replacement=edit["original"]))
        shift += len(edit["replacement"]) - (edit["end"] - edit["start"])
    return dict(text=source, edits=inverse)


def _explicit_identifier(before):
    return bool(context._IDENTIFIER_BEFORE.search(before) or context._NICKNAME_BEFORE.search(before))


def _quantity_prefix(before):
    # Reuse existing quantity labels; only tolerate their grammatical connector.
    return bool(context._QUANTITY_BEFORE.search(before)
                or context._QUANTITY_BEFORE.search(_CONNECTOR.sub("", before)))


def _local_school_context(text, start, end, resources):
    before, after = text[max(0, start - 24):start], text[end:].lstrip(" \t")
    if _explicit_identifier(before):
        return "identifier"
    if _quantity_prefix(before):
        return "quantity"
    if context._MATH_BEFORE.search(before):
        return "uncertain"
    # These are existing adjacent forms, not the old 64-codepoint word window.
    if context._EDUCATION_IDENTITY.match(after):
        return "uncertain" if context._IDENTITY_NONPERSON.match(after) else "local"
    if context._EDUCATION_COMPOUND.match(after):
        return "local"
    if any(after.startswith(term) for term in resources["context"]["project_suffixes"]):
        return "uncertain" if after.startswith("工程师") else "local"
    if any(after.startswith(term) for term in resources["context"]["education_terms"]):
        return "local"
    return "uncertain"


def preprocess_context(text, *, resources):
    """Filter frozen context edits; unresolved codes remain protected for review."""
    base = context.preprocess(text, resources=resources)
    edits, warnings, quantities, withheld = [], list(base["warnings"]), set(), []
    for edit in base["edits"]:
        start, end = edit["start"], edit["end"]
        reason = None
        if edit["reason"] == "read_school_project_identifier":
            decision = _local_school_context(text, start, end, resources)
            if decision != "local":
                reason = "school_" + decision
                if decision == "quantity":
                    quantities.add((start, end))
        elif _explicit_identifier(text[max(0, start - 24):start]):
            reason = "slang_identifier"
        if reason:
            warnings.append(f"needs_review:audit_withdrawn_{reason}:{start}:{end}")
            if (start, end) not in quantities:
                withheld.append((start, end))
        else:
            edits.append(edit)
    output, generated = _rebuild(text, edits)
    source_spans = list(reading.protected_spans(text)) + withheld
    codes = [entry["surface"] for entry in resources["school_codes"]]
    if codes:
        pattern = re.compile(r"(?<![A-Za-z0-9_.])(?:" + "|".join(map(re.escape, codes)) + r")(?![A-Za-z0-9_.])")
        numeric_spans = [match.span() for match in context._NUMBER_EXPRESSION.finditer(text)]
        for match in pattern.finditer(text):
            if (match.span() not in quantities
                    and not context._is_quantity(text, *match.span(), numeric_spans)):
                source_spans.append(match.span())
    result = dict(text=output, edits=edits, warnings=warnings, policy_version=POLICY_VERSION + ":context",
                  historical_policy_version=base["policy_version"])
    spans = semantic.remap_protection(source_spans, text, result) + generated
    for entry in resources["slang"] + resources["school_codes"]:
        spans.extend(match.span() for match in re.finditer(re.escape(entry["spoken_form"]), output))
    result["protected_spans"] = semantic._merge(spans)
    return result


def preprocess_semantic(text, *, resources):
    """Withdraw unsegmented word-tail guesses such as 报名 / 妥当 + a picture."""
    base = semantic.preprocess(text, resources=resources)
    edits, warnings = [], list(base["warnings"])
    for edit in base["edits"]:
        start, end = edit["start"], edit["end"]
        before, after = text[:start].rstrip(" \t"), text[end:].lstrip(" \t")
        explicit = (semantic._COMPARISON.match(after)
                    or (len(before) >= 2 and before[-1] in semantic._CLASSIFIERS
                        and _NUMERAL.search(before[:-1])))
        if explicit:
            edits.append(edit)
        else:
            warnings.append(f"needs_review:audit_withdrawn_word_tail:{start}:{end}")
    output, generated = _rebuild(text, edits)
    # All previously named/retained pictures still require exact protection,
    # including pictures whose naming was withdrawn by this adapter.
    original_spans = semantic.remap_protection(base["protected_spans"], base["text"], _reverse(text, base))
    result = dict(text=output, edits=edits, warnings=warnings, policy_version=POLICY_VERSION + ":semantic",
                  historical_policy_version=base["policy_version"])
    result["protected_spans"] = semantic._merge(
        generated + semantic.remap_protection(original_spans, text, result))
    return result


def preprocess(text, *, resources, semantic_resources, lexicon, context_resources):
    """Compose rules only and return exact protection for offline TN validation.

    ``stages`` edits use each preceding stage's output coordinates; top-level
    protected_spans index final text. No full-sentence reference is accepted.
    """
    lexical = {entry["surface"] for entry in semantic_resources["entries"]}
    stages = {}
    stages["comment"] = comment.preprocess(text, platform_emotes=set(resources["platform_emotes"]),
        unicode_emotes=set(resources["unicode_emotes"]), lexical_emotes=lexical)
    current = stages["comment"]["text"]
    spans = stages["comment"]["semantic_protected_spans"]
    for name, function, kwargs in (
        ("semantic", preprocess_semantic, dict(resources=semantic_resources)),
        ("reading", reading.preprocess, dict(lexicon=lexicon)),
        ("context", preprocess_context, dict(resources=context_resources)),
    ):
        result = function(current, **kwargs)
        replayed, generated = _rebuild(current, result["edits"])
        if replayed != result["text"]:
            raise ValueError("stage edits do not reconstruct " + name)
        spans = semantic.remap_protection(spans, current, result)
        # Reading-generated letters/words were missing from the frozen runner.
        spans = semantic._merge(spans + generated + result["protected_spans"])
        stages[name] = result
        current = result["text"]
    return dict(text=current, stages=stages, protected_spans=spans,
                warnings=[name + ":" + warning for name, stage in stages.items() for warning in stage["warnings"]],
                policy_version=POLICY_VERSION, historical_base=HISTORICAL_BASE)
