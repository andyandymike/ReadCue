"""Externally sourced, whole-token reading hints with auditable edits.

No built-in product dictionary or evaluation reference is consulted. Lexicon
evidence and policy selection belong to the caller and travel with its manifest.
"""

from __future__ import annotations

import re

from readcue.comment_policy import protected_spans as comment_protected_spans


POLICY_VERSION = "reading-policy-v1"
# Preserve identifier punctuation within a token; a final period is punctuation.
# In particular, do not find API inside xAPI, API_extra, API-x or API/unknown.
_TOKEN = re.compile(r"[A-Za-z0-9_+#]+(?:[./-]+[A-Za-z0-9_+#]+)*(?:[-/]+)?")
_LETTERS = re.compile(r"[A-Za-z]+\Z")
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)*")
_DIGITS = "零一二三四五六七八九"
_KINDS = {"initialism", "word", "version_stem", "compound", "ambiguous"}


def validate_lexicon(lexicon: dict) -> None:
    """Validate a lexicon before processing any input; reject conflicting keys."""
    if not isinstance(lexicon, dict) or lexicon.get("schema_version") != 1:
        raise ValueError("lexicon schema_version must be 1")
    if not isinstance(lexicon.get("policy_version"), str) or not lexicon["policy_version"]:
        raise ValueError("lexicon policy_version must be a nonempty string")
    if not lexicon.get("provenance"):
        raise ValueError("lexicon requires provenance")
    if not isinstance(lexicon.get("entries"), list):
        raise ValueError("lexicon entries must be a list")
    identities: set[str] = set()
    surfaces: list[tuple[str, bool]] = []
    for entry in lexicon["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("lexicon entries must be objects")
        identity = entry.get("id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("lexicon entry ids must be unique nonempty strings")
        identities.add(identity)
        kind, action = entry.get("kind"), entry.get("action")
        if kind not in _KINDS:
            raise ValueError(f"unsupported lexicon kind: {kind}")
        expected_action = {"initialism": "spell_letters", "word": "read_word",
                           "compound": "compose", "ambiguous": "preserve_review"}
        if kind in expected_action and action != expected_action[kind]:
            raise ValueError(f"inconsistent action for {identity}")
        if not isinstance(entry.get("spoken_form"), str):
            raise ValueError(f"spoken_form must be a string for {identity}")
        if not entry.get("evidence") or not entry.get("reading_basis"):
            raise ValueError(f"evidence and reading_basis required for {identity}")
        if kind == "version_stem":
            if entry.get("version_policy") != "integer_components":
                raise ValueError(f"unsupported version_policy for {identity}")
            base_action = entry.get("base_action")
            if base_action not in {"letters", "word"}:
                raise ValueError(f"unsupported base_action for {identity}")
            if action != {"letters": "spell_letters", "word": "read_word"}[base_action]:
                raise ValueError(f"inconsistent version action for {identity}")
        components = entry.get("components")
        if kind == "compound":
            if not isinstance(components, list) or not components:
                raise ValueError(f"compound components required for {identity}")
            for component in components:
                if (not isinstance(component, dict)
                        or not isinstance(component.get("text"), str)
                        or not _LETTERS.fullmatch(component["text"])
                        or component.get("action") not in {"read_word", "spell_letters"}):
                    raise ValueError(f"invalid compound component for {identity}")
        aliases = entry.get("surfaces")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"surfaces required for {identity}")
        for surface in aliases:
            if (not isinstance(surface, dict) or not isinstance(surface.get("text"), str)
                    or not _LETTERS.fullmatch(surface["text"])
                    or type(surface.get("case_sensitive")) is not bool):
                raise ValueError(f"surface requires ASCII letters and case_sensitive for {identity}")
            value, sensitive = surface["text"], surface["case_sensitive"]
            for previous, previous_sensitive in surfaces:
                if ((sensitive and previous_sensitive and value == previous)
                        or ((not sensitive or not previous_sensitive)
                            and value.lower() == previous.lower())):
                    raise ValueError(f"conflicting lexicon surface: {value}")
            surfaces.append((value, sensitive))
            if kind == "compound" and value.lower() != "".join(
                    component["text"] for component in components).lower():
                raise ValueError(f"compound components do not match surface for {identity}")


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((start, end))
    return result


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Protect v1 literals and complete mixed letter/digit identifiers from TN.

    Pure numbers remain available to TN. Pure English is not additionally
    protected; existing code, quote, mention and URL protection still applies.
    """
    spans = comment_protected_spans(text)
    for match in _TOKEN.finditer(text):
        if re.search(r"[A-Za-z]", match[0]) and re.search(r"[0-9]", match[0]):
            spans.append(match.span())
    return _merge(spans)


def _matches(value: str, surface: dict) -> bool:
    expected = surface["text"]
    return value == expected if surface["case_sensitive"] else value.lower() == expected.lower()


def _letters(value: str) -> str:
    return " ".join(value.upper())


def _render_entry(value: str, entry: dict) -> str:
    if entry["action"] == "spell_letters":
        return _letters(value)
    if entry["action"] == "compose":
        chunks: list[str] = []
        offset = 0
        for component in entry["components"]:
            part = value[offset:offset + len(component["text"])]
            chunks.append(_letters(part) if component["action"] == "spell_letters" else part)
            offset += len(component["text"])
        return " ".join(chunks)
    return value


def _four_digits(value: int) -> str:
    result = ""
    pending_zero = False
    for divisor, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        digit, value = divmod(value, divisor)
        if digit:
            if pending_zero:
                result += "零"
            result += _DIGITS[digit] + unit
            pending_zero = False
        elif result and value:
            pending_zero = True
    return result


def _integer(value: str) -> str | None:
    if len(value) > 1 and value.startswith("0"):
        return "".join(_DIGITS[int(digit)] for digit in value)
    if value == "0":
        return "零"
    # Larger unpadded values are retained for review, rather than silently read
    # digit-by-digit contrary to the selected integer-components convention.
    if len(value) > 16:
        return None
    groups = [int(value[max(0, end - 4):end]) for end in range(len(value), 0, -4)]
    result = ""
    skipped = False
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not group:
            skipped = bool(result)
            continue
        if result and (skipped or group < 1000):
            result += "零"
        result += _four_digits(group) + ("", "万", "亿", "兆")[index]
        skipped = False
    return result[1:] if result.startswith("一十") else result


def _version(value: str) -> str | None:
    numbers = [_integer(component) for component in value.split(".")]
    return None if any(number is None for number in numbers) else "点".join(numbers)


def _render_version_token(value: str, entries: list[dict]) -> tuple[str, list[str]] | None:
    """Parse every stem+version component; unknown suffixes invalidate all edits."""
    stems = [(surface, entry) for entry in entries if entry["kind"] == "version_stem"
             for surface in entry["surfaces"]]
    stems.sort(key=lambda item: (-len(item[0]["text"]), item[0]["text"]))
    memo: dict[int, list[tuple[str, str]] | None] = {}

    def parse(position: int) -> list[tuple[str, str]] | None:
        if position == len(value):
            return []
        if position in memo:
            return memo[position]
        for surface, entry in stems:
            stem_end = position + len(surface["text"])
            if not _matches(value[position:stem_end], surface):
                continue
            match = _VERSION.match(value, stem_end)
            if match is None:
                continue
            reading = _version(match[0])
            if reading is None:
                continue
            remainder = parse(match.end())
            if remainder is not None:
                result = [(_render_entry(value[position:stem_end], entry) + " " + reading,
                           entry["id"])] + remainder
                memo[position] = result
                return result
        memo[position] = None
        return None

    result = parse(0)
    return (" ".join(part for part, _ in result), [identity for _, identity in result]) if result else None


def preprocess(text: str, *, lexicon: dict) -> dict:
    """Return text, original-code-point edits, warnings and output protection.

    The optional ``protected_spans`` result uses coordinates of the returned
    text, suitable for the subsequent TN stage. Edits always use input offsets.
    Lexicon ``spoken_form`` records rationale; rendering follows explicit
    actions and actual input spelling, never arbitrary free-text substitution.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    validate_lexicon(lexicon)
    literal = comment_protected_spans(text)
    entries = lexicon["entries"]
    edits: list[dict] = []
    warnings: list[str] = []
    for match in _TOKEN.finditer(text):
        start, end = match.span()
        if any(start < right and left < end for left, right in literal):
            continue
        value = match[0]
        if not re.search(r"[A-Za-z]", value):
            continue
        entry = next((entry for entry in entries
                      if any(_matches(value, surface) for surface in entry["surfaces"])), None)
        if entry is not None and entry["kind"] == "ambiguous":
            warnings.append(f"needs_review:ambiguous_reading:{entry['id']}:{start}:{end}")
            continue
        if entry is not None:
            replacement = _render_entry(value, entry)
            reason = f"lexicon_{entry['action']}"
            entry_ids = [entry["id"]]
        else:
            version_result = _render_version_token(value, entries)
            reason = "lexicon_version_components"
            if version_result is None:
                warnings.append(f"needs_review:unknown_ascii_token:{start}:{end}")
                continue
            replacement, entry_ids = version_result
        if replacement != value:
            edits.append({"start": start, "end": end, "original": value,
                          "replacement": replacement, "reason": reason,
                          "lexicon_entry_ids": entry_ids})
    chunks: list[str] = []
    cursor = 0
    for edit in edits:
        chunks.extend((text[cursor:edit["start"]], edit["replacement"]))
        cursor = edit["end"]
    chunks.append(text[cursor:])
    output = "".join(chunks)
    return {"text": output, "edits": edits, "warnings": warnings,
            "policy_version": POLICY_VERSION, "lexicon_policy_version": lexicon["policy_version"],
            "protected_spans": protected_spans(output)}
