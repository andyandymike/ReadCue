"""Comment preprocessing with conservative protection of picture word slots.

All removable emotes must be supplied by the caller from independently recorded
sources. This module has no sample-specific dictionary and never accepts labels.
It does not name emoji or normalize whitespace, words or numbers. Deletion edit
offsets retain the v1 input-coordinate contract; semantic protection offsets
refer to the returned text, after reply/emotion removals and boundary spaces.
"""

from __future__ import annotations

import re
import unicodedata


POLICY_VERSION = "comment-policy-v2"

_REPLY = re.compile(r"\A[ \t]*回复[ \t]*@[^\r\n\v\f\x1c-\x1e\x85\u2028\u2029:：]+[:：][ \t]*")
_URL = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+.-]*://|www\.|mailto:)[^\s<>\"'“”‘’「」『』]+",
    re.IGNORECASE,
)
_MENTION = re.compile(r"(?<![A-Za-z0-9_.])@[^\s@:：,，。;；!?！？<>\"'“”‘’「」『』]+")
_QUOTE_PATTERNS = (
    re.compile(r'"(?:\\.|[^"\\])*"'),
    re.compile(r"(?<![A-Za-z0-9])'(?:\\.|[^'\\])*'(?![A-Za-z0-9])"),
    re.compile(r"“[^”]*”"),
    re.compile(r"‘[^’]*’"),
    re.compile(r"「[^」]*」"),
    re.compile(r"『[^』]*』"),
)
# Syntactic caution only: a nearby classifier can give a picture noun meaning.
_CLASSIFIERS = frozenset("个只把颗条朵枚粒根块张片瓶杯台件辆本双对头匹座支株艘盏团串份包盒罐盘碗套扇位名")
_PREDICATE_ENDINGS = ("是", "像", "当", "成为", "变成")
_COMPARISON_STARTS = ("一样", "一般", "似的", "似地", "般的", "般地")


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < right and left < end for left, right in spans)


def _code_spans(text: str) -> list[tuple[int, int]]:
    """Protect matching backtick runs; an unclosed run protects the remainder."""
    runs = list(re.finditer(r"`+", text))
    result: list[tuple[int, int]] = []
    index = 0
    while index < len(runs):
        opening = runs[index]
        closing = index + 1
        while closing < len(runs) and len(runs[closing][0]) != len(opening[0]):
            closing += 1
        if closing == len(runs):
            result.append((opening.start(), len(text)))
            break
        result.append((opening.start(), runs[closing].end()))
        index = closing + 1
    return result


def _bracket_spans(text: str) -> list[tuple[int, int]]:
    """Return outer bracket expressions, including an unfinished expression."""
    result: list[tuple[int, int]] = []
    start = None
    depth = 0
    for index, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]" and depth:
            depth -= 1
            if depth == 0:
                result.append((start, index + 1))
                start = None
    if start is not None:
        result.append((start, len(text)))
    return result


def _literal_spans(text: str) -> list[tuple[int, int]]:
    spans = _code_spans(text)
    spans.extend(match.span() for match in _URL.finditer(text))
    for pattern in _QUOTE_PATTERNS:
        spans.extend(match.span() for match in pattern.finditer(text))
    return _merge(spans)


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Return merged half-open spans for a downstream model-edit guard.

    Code, URLs, body mentions, paired quotes and bracket expressions are protected. Call
    this on the *preprocessed* text: approved platform tokens have already been
    removed by then, while the remaining brackets are literal or unknown.
    """
    return _merge(_literal_spans(text) + _bracket_spans(text)
                  + [match.span() for match in _MENTION.finditer(text)])


def _is_extension(char: str) -> bool:
    number = ord(char)
    return (
        unicodedata.category(char).startswith("M")
        or 0x1F3FB <= number <= 0x1F3FF
        or 0xE0020 <= number <= 0xE007F
    )


def _is_regional_indicator(char: str) -> bool:
    return 0x1F1E6 <= ord(char) <= 0x1F1FF


def _sequence_boundaries(text: str) -> set[int]:
    """Conservative boundaries for emoji bases, modifiers, flags and ZWJ chains.

    This is deliberately not advertised as a general Unicode grapheme parser.
    Combining marks, variation selectors and tag characters stay attached, so
    an allowed component cannot be deleted from an unlisted larger sequence.
    """
    boundaries = {0}
    position = 0
    while position < len(text):
        first = text[position]
        position += 1
        if first == "\u200d" and position < len(text):
            # Preserve a malformed leading joiner together with its base too.
            position += 1
        if (_is_regional_indicator(first) and position < len(text)
                and _is_regional_indicator(text[position])):
            position += 1
        while position < len(text) and _is_extension(text[position]):
            position += 1
        while position < len(text) and text[position] == "\u200d":
            position += 1
            if position < len(text):
                position += 1
            while position < len(text) and _is_extension(text[position]):
                position += 1
        boundaries.add(position)
    return boundaries


def _classifier_before(text: str, start: int) -> bool:
    preceding = text[:start].rstrip(" \t")
    return bool(preceding and preceding[-1] in _CLASSIFIERS)


def _ascii_alphanumeric(char: str) -> bool:
    return char.isascii() and char.isalnum()


def _picture_sequence(value: str) -> bool:
    """Recognize candidates for preservation, never an authority to delete.

    Broad pictographic/symbol blocks intentionally include uncertain symbols.
    This is not a Unicode Emoji property implementation; complete unlisted
    clusters stay intact and require review instead of being guessed as words.
    """
    return any(
        0x1F000 <= ord(char) <= 0x1FAFF
        or 0x2300 <= ord(char) <= 0x23FF
        or 0x2600 <= ord(char) <= 0x27BF
        or char in "\u00a9\u00ae\u203c\u2049\u2122\u2139\u20e3\ufe0f"
        for char in value
    )


def _semantic_reason(text: str, start: int, end: int, *, lexical: bool) -> str | None:
    if _classifier_before(text, start):
        return "classifier_before_emote"
    if text[:start].rstrip(" \t").endswith(_PREDICATE_ENDINGS):
        return "predicate_before_emote"
    following = text[end:].lstrip(" \t")
    if following.startswith(_COMPARISON_STARTS):
        return "comparison_after_emote"
    # An independently sourced object list can support a noun modifier slot.
    # Bare 的 is not a trigger for emotional faces (e.g. 好的😊的确如此).
    if lexical and re.match(r"的(?!确)[\u3400-\u9fff]", following):
        return "lexical_noun_modifier"
    if lexical:
        return "lexical_emote"
    return None


def preprocess(
    text: str, *, platform_emotes: set[str], unicode_emotes: set[str],
    lexical_emotes: set[str] | None = None,
) -> dict:
    """Apply reply/emotion policy without deleting potential picture words.

    ``platform_emotes`` contains bare names (without ``[`` and ``]``).
    ``unicode_emotes`` contains complete removable sequences, selected by the
    caller. ``lexical_emotes`` contains independently sourced complete Unicode
    object/picture sequences, not their names or a pronunciation dictionary.
    Lexical membership overrides removal membership. Syntax can also retain a
    removable emote in a noun/comparison slot. Unknown picture clusters outside
    code, quotes, URLs, mentions and brackets are retained and protected.

    ``edits`` and warning suffixes use original input code-point offsets.
    ``semantic_protected_spans`` and annotation ``start``/``end`` use returned
    text offsets; annotation ``original_start``/``original_end`` point back to
    the input. All intervals are half-open, not UTF-16 or byte offsets. Only
    annotations with ``contextual_noun_slot`` have an explicit syntactic cue;
    membership alone remains pending, and no emoji is given a spoken name here.

    Reply-wrapper deletion includes horizontal whitespace immediately after the
    colon but never a newline. Other spaces and punctuation are preserved.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if any(not isinstance(name, str) or not name or any(c in name for c in "[]\r\n")
           for name in platform_emotes):
        raise ValueError("platform_emotes must contain nonempty bare token names")
    if any(not isinstance(emote, str) or not emote for emote in unicode_emotes):
        raise ValueError("unicode_emotes must contain nonempty complete sequences")
    lexical_emotes = set() if lexical_emotes is None else lexical_emotes
    if any(not isinstance(emote, str) or not emote for emote in lexical_emotes):
        raise ValueError("lexical_emotes must contain nonempty complete sequences")

    edits: list[dict] = []
    warnings: list[str] = []
    semantic: list[dict] = []
    literal = _literal_spans(text)
    prefix = _REPLY.match(text)
    if prefix and not _overlaps(*prefix.span(), literal):
        nickname = prefix[0].split("@", 1)[1].rsplit(":" if ":" in prefix[0] else "：", 1)[0]
        if nickname.strip(" \t"):
            edits.append({"start": 0, "end": prefix.end(), "original": prefix[0],
                          "replacement": "", "reason": "omit_reply_prefix"})

    # Mentions are literal after wrapper recognition. Including them earlier
    # would suppress every real reply prefix because its nickname begins @.
    protected = list(literal) + [match.span() for match in _MENTION.finditer(text)]
    if edits:
        protected.append((edits[0]["start"], edits[0]["end"]))
    platform_candidates: list[tuple[int, int]] = []
    for start, end in _bracket_spans(text):
        value = text[start:end]
        if value.endswith("]") and value[1:-1] in platform_emotes:
            platform_candidates.append((start, end))
        else:
            protected.append((start, end))
    protected = _merge(protected)

    def add_emote(start: int, end: int, reason: str, *, unknown: bool = False) -> None:
        if _overlaps(start, end, protected):
            return
        lexical = text[start:end] in lexical_emotes
        keep_reason = _semantic_reason(text, start, end, lexical=lexical)
        if keep_reason is None and unknown:
            keep_reason = "unknown_unicode_emote"
        if keep_reason is not None:
            warnings.append(f"needs_review:{keep_reason}:{start}:{end}")
            semantic.append({"original_start": start, "original_end": end,
                             "surface": text[start:end], "reason": keep_reason,
                             "in_lexical_emotes": lexical,
                             "contextual_noun_slot": keep_reason not in {
                                 "lexical_emote", "unknown_unicode_emote"},
                             "needs_review": True})
            return
        edits.append({"start": start, "end": end, "original": text[start:end],
                      "replacement": "", "reason": reason})

    for start, end in platform_candidates:
        add_emote(start, end, "omit_platform_emote")
        # Never process a Unicode component inside a platform token separately.
        protected.append((start, end))
    protected = _merge(protected)

    boundaries = _sequence_boundaries(text)
    ordered_unicode = sorted(unicode_emotes | lexical_emotes,
                             key=lambda item: (-len(item), item))
    next_boundary = dict(zip(sorted(boundaries), sorted(boundaries)[1:]))
    position = 0
    while position < len(text):
        if position in boundaries:
            matched = False
            for emote in ordered_unicode:
                end = position + len(emote)
                if end in boundaries and text.startswith(emote, position):
                    add_emote(position, end, "omit_unicode_emote")
                    position = end - 1
                    matched = True
                    break
            if not matched:
                end = next_boundary[position]
                if _picture_sequence(text[position:end]):
                    add_emote(position, end, "unknown_unicode_emote", unknown=True)
                position = end - 1
        position += 1

    edits.sort(key=lambda edit: edit["start"])
    # Consecutive removals form one boundary: A[smile][smile]B must not become AB.
    index = 0
    while index < len(edits):
        end_index = index + 1
        while (end_index < len(edits)
               and edits[end_index - 1]["end"] == edits[end_index]["start"]):
            end_index += 1
        start, end = edits[index]["start"], edits[end_index - 1]["end"]
        if (start > 0 and end < len(text) and _ascii_alphanumeric(text[start - 1])
                and _ascii_alphanumeric(text[end])):
            edits[index]["replacement"] = " "
        index = end_index

    chunks: list[str] = []
    cursor = 0
    for edit in edits:
        if edit["start"] < cursor:
            raise AssertionError("policy edits must not overlap")
        chunks.extend((text[cursor:edit["start"]], edit["replacement"]))
        cursor = edit["end"]
    chunks.append(text[cursor:])
    output = "".join(chunks)
    semantic.sort(key=lambda item: item["original_start"])
    for item in semantic:
        start, end = item["original_start"], item["original_end"]
        if any(start < edit["end"] and edit["start"] < end for edit in edits):
            raise AssertionError("semantic pictures must not overlap deletions")
        shift = sum(len(edit["replacement"]) - (edit["end"] - edit["start"])
                    for edit in edits if edit["end"] <= start)
        item["start"], item["end"] = start + shift, end + shift
        if output[item["start"]:item["end"]] != item["surface"]:
            raise AssertionError("semantic output coordinates must preserve surface")
    return {"text": output, "edits": edits, "warnings": warnings,
            "semantic_protected_spans": _merge([
                (item["start"], item["end"]) for item in semantic]),
            "semantic_annotations": semantic, "policy_version": POLICY_VERSION}
