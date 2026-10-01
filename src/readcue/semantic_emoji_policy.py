"""Render independently sourced pictograph names only in explicit noun slots."""
from __future__ import annotations

import re
from readcue.comment_policy import _sequence_boundaries, _merge
from readcue.reading_policy import protected_spans as literal_spans

POLICY_VERSION = "semantic-emoji-v1"
_CLASSIFIERS = frozenset("个只把颗条朵枚粒根块张片瓶杯台件辆本双对头匹座支株艘盏团串份包盒罐盘碗套扇位名")
_PREDICATES = ("是", "像", "当", "成为", "变成")
_COMPARISON = re.compile(r"(?:一样|一般|似的|般的)")


def validate_resources(resources):
    if not isinstance(resources, dict) or resources.get("schema_version") != 1:
        raise ValueError("semantic emoji resources require schema_version 1")
    if not resources.get("policy_version") or not resources.get("provenance"):
        raise ValueError("semantic emoji resource provenance is required")
    entries = resources.get("entries")
    if not isinstance(entries, list):
        raise ValueError("semantic emoji entries must be a list")
    ids, surfaces = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("semantic emoji entry must be an object")
        identity, surface, spoken = entry.get("id"), entry.get("surface"), entry.get("spoken_form")
        if not all(isinstance(v, str) and v for v in (identity, surface, spoken)) or not entry.get("evidence"):
            raise ValueError("semantic emoji entry requires id, surface, spoken_form and evidence")
        if identity in ids or surface in surfaces or any(c.isspace() for c in surface):
            raise ValueError("duplicate or invalid semantic emoji surface")
        if not all("\u3400" <= c <= "\u9fff" for c in spoken):
            raise ValueError("semantic emoji spoken forms must be Chinese noun names")
        def picture_component(c):
            value = ord(c)
            return (0x1F000 <= value <= 0x1FAFF or 0x2600 <= value <= 0x27BF
                    or value in (0x200D, 0xFE0E, 0xFE0F) or 0xE0020 <= value <= 0xE007F)
        if (_sequence_boundaries(surface) != {0, len(surface)}
                or not all(picture_component(c) for c in surface)
                or not any(0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF for c in surface)):
            raise ValueError("semantic emoji surfaces must be a single complete picture sequence without ordinary text")
        ids.add(identity); surfaces.add(surface)


def remap_protection(spans, source, result):
    """Carry output-coordinate protection across a verified, nonoverlapping stage.

    Exact replacements are allowed and their whole replacement remains protected.
    Partly overlapping edits are rejected, so a later policy cannot silently cut
    an unknown emoji/identifier in half or discard part of a protected phrase.
    """
    remapped = []
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError("invalid protected span")
        start, end = span
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source):
            raise ValueError("protected span outside its source")
        a, b = start, end
        for edit in result["edits"]:
            left, right = edit["start"], edit["end"]
            delta = len(edit["replacement"]) - (right - left)
            if right <= start:
                a += delta; b += delta
            elif left >= end:
                continue
            elif start <= left and right <= end:
                if not edit["replacement"]:
                    raise ValueError("a later stage tried to remove protected content")
                b += delta
            else:
                raise ValueError("a later stage partly overlaps protected content")
        if not 0 <= a < b <= len(result["text"]):
            raise ValueError("remapped protection outside output")
        remapped.append((a, b))
    return _merge(remapped)


def preprocess(text, *, resources):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    validate_resources(resources)
    literals = literal_spans(text)
    boundaries = _sequence_boundaries(text)
    edits, warnings, retained = [], [], []
    consumed = set()
    for entry in sorted(resources["entries"], key=lambda e: (-len(e["surface"]), e["surface"])):
        for match in re.finditer(re.escape(entry["surface"]), text):
            start, end = match.span()
            # A presentation selector belongs to the same pictograph. Other
            # modifiers / ZWJ chains must not be converted component by component.
            if end < len(text) and text[end] in ("\ufe0e", "\ufe0f"):
                end += 1
            if start not in boundaries or end not in boundaries or start in consumed:
                continue
            consumed.add(start)
            if any(start < b and a < end for a, b in literals):
                continue
            before, after = text[:start].rstrip(" \t"), text[end:].lstrip(" \t")
            noun_slot = bool((before and before[-1] in _CLASSIFIERS)
                             or before.endswith(_PREDICATES) or _COMPARISON.match(after))
            if noun_slot:
                edits.append({"start": start, "end": end, "original": text[start:end],
                              "replacement": entry["spoken_form"], "reason": "read_sourced_pictograph_in_noun_slot",
                              "resource_entry_id": entry["id"], "evidence": entry["evidence"]})
            else:
                retained.append((start, end))
                warnings.append(f"needs_review:semantic_emoji_context:{start}:{end}")
    edits.sort(key=lambda e: e["start"])
    cursor, parts, generated, shift = 0, [], [], 0
    for edit in edits:
        if edit["start"] < cursor:
            raise ValueError("overlapping semantic emoji edits")
        parts.extend([text[cursor:edit["start"]], edit["replacement"]])
        left = edit["start"] + shift
        generated.append((left, left + len(edit["replacement"])))
        shift += len(edit["replacement"]) - (edit["end"] - edit["start"])
        cursor = edit["end"]
    output = "".join(parts) + text[cursor:]
    result = {"text": output, "edits": edits, "warnings": warnings, "policy_version": POLICY_VERSION,
              "resources_policy_version": resources["policy_version"]}
    result["protected_spans"] = _merge(generated + remap_protection(retained, text, result))
    return result
