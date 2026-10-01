"""Bounded slang and school-project reading policy, isolated from gold labels."""

from __future__ import annotations

import re

from readcue.comment_policy import protected_spans as literal_spans
from readcue.reading_policy import protected_spans as reading_protected_spans


POLICY_VERSION = "context-reading-v1"
_TOKEN = re.compile(r"[A-Za-z0-9_+#]+(?:[./-]+[A-Za-z0-9_+#]+)*(?:[-/]+)?")
_NUMBER_EXPRESSION = re.compile(r"[0-9]+(?:[ \t]*[.,:/\-−×xX*+][ \t]*[0-9]+)+")
_QUANTITY_AFTER = re.compile(
    r"[ \t]*(?:万|亿|千|百|元|块|角|分|美金|美元|日元|人民币|[%％‰]|度|℃|°|"
    r"名|所|位|个|人|家|台|只|件|条|张|本|份|套|支|辆|座|门|节|门课|倍|"
    r"公里|厘米|毫米|米|平方|立方|公斤|千克|克|斤|吨|升|毫升|英寸|寸|像素|"
    r"路|号|室|楼|层|页|级|届|年|月|日|天|小时|分钟|秒|岁|周|次|票|班|"
    r"kg\b|km\b|cm\b|mm\b|px\b|g\b|m\b|GB\b|MB\b)", re.IGNORECASE,
)
_QUANTITY_BEFORE = re.compile(
    r"(?:第|金额|费用|学费|预算|价格|成本|总价|收费|收取|花费|支付|长|宽|高|"
    r"长度|宽度|高度|尺寸|面积|体积|重量|数量|人数|总数|总计|合计|共计|"
    r"排名|排第|编号|门牌|房号|房间号|页码|序号|订单号|航班号|[¥￥$])"
    r"[ \t]*[:：=]?[ \t]*$"
)
_IDENTIFIER_BEFORE = re.compile(
    r"(?:变量|字段|参数|键名|账号|用户名|标识|标识符|型号|文件名|密码|编号|ID)"
    r"(?:的值|值|名称|名)?[ \t]*(?:是|为|叫做|叫|设为)?"
    r"[ \t]*[:：=]?[ \t]*$", re.IGNORECASE,
)
_MATH_BEFORE = re.compile(
    r"(?:计算|算式|算出|求值|比值|比例|分数|加法|减法|乘法|除法|数学题|数学作业)"
    r"[ \t:：=]*$"
)


def validate_resources(resources: dict) -> None:
    if not isinstance(resources, dict) or resources.get("schema_version") != 1:
        raise ValueError("context resources schema_version must be 1")
    if not isinstance(resources.get("policy_version"), str) or not resources["policy_version"]:
        raise ValueError("context resources policy_version is required")
    provenance = resources.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("context resources require primary-source provenance")
    source_ids = set()
    for source in provenance:
        if (not isinstance(source, dict) or not isinstance(source.get("id"), str)
                or not source["id"] or source["id"] in source_ids
                or not isinstance(source.get("url"), str) or not source["url"].startswith("https://")):
            raise ValueError("invalid or duplicate context source")
        source_ids.add(source["id"])
    identities, all_surfaces = set(), set()
    for kind in ("slang", "school_codes"):
        entries = resources.get(kind)
        if not isinstance(entries, list):
            raise ValueError(f"context resources require {kind}")
        surfaces = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("context entries must be objects")
            identity, surface = entry.get("id"), entry.get("surface")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("context entry ids must be unique")
            identities.add(identity)
            pattern = r"[A-Za-z0-9]+" if kind == "slang" else r"[0-9]+"
            if not isinstance(surface, str) or not re.fullmatch(pattern, surface) or surface.lower() in surfaces:
                raise ValueError("invalid or duplicate context surface")
            if surface.lower() in all_surfaces:
                raise ValueError("context surfaces must not overlap across entry kinds")
            surfaces.add(surface.lower())
            all_surfaces.add(surface.lower())
            if kind == "slang" and type(entry.get("case_sensitive")) is not bool:
                raise ValueError("slang requires explicit case_sensitive")
            if not isinstance(entry.get("spoken_form"), str) or not entry["spoken_form"]:
                raise ValueError("context spoken_form must be nonempty")
            if not isinstance(entry.get("evidence"), list) or not entry["evidence"] or any(
                    identity not in source_ids for identity in entry["evidence"]):
                raise ValueError("context evidence must identify a captured source")
            if not isinstance(entry.get("reading_basis"), str) or not entry["reading_basis"]:
                raise ValueError("context reading_basis is required")
    context = resources.get("context")
    if not isinstance(context, dict) or type(context.get("window_codepoints")) is not int:
        raise ValueError("context window_codepoints must be an integer")
    if not 1 <= context["window_codepoints"] <= 128:
        raise ValueError("context window must be bounded to 1..128 code points")
    for key in ("education_terms", "project_suffixes"):
        if (not isinstance(context.get(key), list) or not context[key]
                or any(not isinstance(item, str) or not item for item in context[key])):
            raise ValueError(f"context requires nonempty {key}")


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((start, end))
    return result


def _overlap(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < right and left < end for left, right in spans)


def _masked(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            if chars[index] not in "\r\n\u2028\u2029":
                chars[index] = " "
    return "".join(chars)


def _is_quantity(text: str, start: int, end: int, numeric_spans: list[tuple[int, int]]) -> bool:
    if _overlap(start, end, numeric_spans):
        return True
    if _QUANTITY_AFTER.match(text[end:]) or _QUANTITY_BEFORE.search(text[max(0, start - 24):start]):
        return True
    # Full-width decimal/percentage syntax and arithmetic operators are not
    # evidence that a nearby school word turns a quantity into a project name.
    before = text[:start].rstrip(" \t")
    after = text[end:].lstrip(" \t")
    return bool((before and before[-1] in "+-*＋－−×÷＝=￥$¥")
                or (after and after[0] in "+-*＋－−×÷＝="))


def _education_context(text: str, start: int, end: int, context: dict, codes: set[str]) -> bool:
    suffix = text[end:].lstrip(" \t")
    if any(suffix.startswith(term) for term in context["project_suffixes"]) and not suffix.startswith("工程师"):
        return True
    width = context["window_codepoints"]
    left, right = max(0, start - width), min(len(text), end + width)
    # A nearby next sentence can resolve an elliptical remark, but unrelated
    # paragraphs and protected quotations cannot supply a school context.
    before = re.split(r"[\r\n\u2028\u2029]", text[left:start])[-1]
    after = re.split(r"[\r\n\u2028\u2029]", text[end:right])[0]
    if any(term in before or term in after for term in context["education_terms"]):
        return True
    # An explicitly named neighboring school project also disambiguates a list
    # such as code-A and code-B工程, without treating every 工程 as education.
    for code in codes:
        for term in context["project_suffixes"]:
            pattern = re.escape(code) + r"[ \t]*" + re.escape(term) + r"(?!师)"
            if re.search(pattern, before) or re.search(pattern, after):
                return True
    return False


def preprocess(text: str, *, resources: dict) -> dict:
    """Apply explicit context readings without normalizing unrelated text.

    ``edits`` use original input code-point offsets. ``protected_spans`` use
    returned-text offsets, include all generated readings and ambiguous school
    codes, and must be passed to the subsequent model's edit guard.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    validate_resources(resources)
    literals = literal_spans(text)
    identifiers = reading_protected_spans(text)
    context_text = _masked(text, literals)
    edits, warnings, preserved = [], [], []

    def add(start: int, end: int, entry: dict, reason: str) -> None:
        edits.append({"start": start, "end": end, "original": text[start:end],
                      "replacement": entry["spoken_form"], "reason": reason,
                      "resource_entry_id": entry["id"], "evidence": list(entry["evidence"])})

    for match in _TOKEN.finditer(text):
        start, end = match.span()
        if _overlap(start, end, literals):
            continue
        entry = next((entry for entry in resources["slang"]
                      if (match[0] == entry["surface"] if entry["case_sensitive"]
                          else match[0].lower() == entry["surface"].lower())), None)
        if entry is None:
            continue
        after = text[end:].lstrip(" \t")
        if (_IDENTIFIER_BEFORE.search(text[max(0, start - 24):start])
                or after.startswith(("=", "＝", "(", "()", "的值", "字段", "变量"))):
            warnings.append(f"needs_review:slang_identifier_context:{start}:{end}")
            preserved.append((start, end))
        else:
            add(start, end, entry, "expand_sourced_slang")

    codes = {entry["surface"]: entry for entry in resources["school_codes"]}
    numeric_spans = []
    for match in _NUMBER_EXPRESSION.finditer(text):
        # A slash-separated list of precisely the known project codes may be
        # resolved using school context; all other numeric expressions are TN.
        parts = [part.strip(" \t") for part in match[0].split("/")]
        arithmetic = (_MATH_BEFORE.search(text[max(0, match.start() - 24):match.start()])
                      or text[match.end():].lstrip(" \t").startswith(("的结果", "等于", "=", "＝")))
        if arithmetic or len(parts) < 2 or any(part not in codes for part in parts):
            numeric_spans.append(match.span())
    if codes:
        code_pattern = re.compile(r"(?<![A-Za-z0-9_.])(?:" + "|".join(
            re.escape(value) for value in sorted(codes, key=len, reverse=True)) + r")(?![A-Za-z0-9_.])")
        for match in code_pattern.finditer(text):
            start, end = match.span()
            if _overlap(start, end, identifiers) or _is_quantity(text, start, end, numeric_spans):
                continue
            if _education_context(context_text, start, end, resources["context"], set(codes)):
                add(start, end, codes[match[0]], "read_school_project_identifier")
            else:
                warnings.append(f"needs_review:school_code_context:{start}:{end}")
                preserved.append((start, end))

    edits.sort(key=lambda edit: edit["start"])
    chunks, generated = [], []
    cursor = 0
    output_length = 0
    for edit in edits:
        if edit["start"] < cursor:
            raise AssertionError("context edits must not overlap")
        unchanged = text[cursor:edit["start"]]
        chunks.extend((unchanged, edit["replacement"]))
        output_length += len(unchanged)
        generated.append((output_length, output_length + len(edit["replacement"])))
        output_length += len(edit["replacement"])
        cursor = edit["end"]
    chunks.append(text[cursor:])
    output = "".join(chunks)
    protected = reading_protected_spans(output) + generated
    for start, end in preserved:
        shift = sum(len(edit["replacement"]) - (edit["end"] - edit["start"])
                    for edit in edits if edit["end"] <= start)
        protected.append((start + shift, end + shift))
    # Re-processing an already-expanded phrase must not make it writable by TN.
    for entry in resources["slang"] + resources["school_codes"]:
        protected.extend(match.span() for match in re.finditer(re.escape(entry["spoken_form"]), output))
    return {"text": output, "edits": edits, "warnings": warnings, "policy_version": POLICY_VERSION,
            "resources_policy_version": resources["policy_version"], "protected_spans": _merge(protected)}
