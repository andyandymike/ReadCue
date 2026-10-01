"""Capture independent primary sources for the bounded context policy.

Only source documents and manually declared policy facts are used. No comments,
evaluation cases, human references or previous predictions are read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


SOURCES = [
    {"id": "u1s1_research", "url": "https://hyxx.ybu.edu.cn/__local/1/98/B6/251F27882920B65E214E391507B_88DF3882_C1FD18.pdf",
     "filename": "u1s1-yang-2024.pdf", "title": "“有一说一”的坦言标记功能及其形成",
     "author": "杨万成", "publisher": "汉语学习（延边大学期刊网站）", "date": "2024-08",
     "source_type": "original_research", "locator": "第110页，例⑨之后",
     "supports": "u1s1 is a letter-and-number homophonic form of 有一说一; it does not establish a general rule for arbitrary identifiers."},
    {"id": "moe_985", "url": "https://www.moe.gov.cn/srcsite/A22/s7065/200612/t20061206_128833.html",
     "filename": "moe-985.html", "title": "“985工程”学校名单", "publisher": "中华人民共和国教育部",
     "date": "2006-12-06", "source_type": "official_primary_document",
     "supports": "985 is part of an official higher-education project name, not a count of 985 schools; no official pronunciation is asserted."},
    {"id": "moe_211", "url": "https://www.moe.gov.cn/srcsite/A22/s7065/200512/t20051223_82762.html",
     "filename": "moe-211.html", "title": "“211工程”学校名单", "publisher": "中华人民共和国教育部",
     "date": "2005-12-23", "source_type": "official_primary_document",
     "supports": "211 is part of an official higher-education project name, not a count of 211 schools; 二幺幺 is the user's reading convention, not an official unique pronunciation."},
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite resource capture: {output}")
    output.mkdir(parents=True)
    captured = []
    for source in SOURCES:
        request = Request(source["url"], headers={"User-Agent": "ReadCue/0.1 primary-source research"})
        with urlopen(request, timeout=45) as response:
            raw = response.read(24 * 1024 * 1024 + 1)
            if len(raw) > 24 * 1024 * 1024:
                raise ValueError("Source exceeds bounded capture size")
            headers = dict(response.headers.items())
            final_url = response.url
        if source["filename"].endswith(".pdf") and not raw.startswith(b"%PDF"):
            raise ValueError("Research PDF response is not a PDF")
        if source["filename"].endswith(".html"):
            decoded = raw.decode("utf-8", errors="ignore")
            code = "985" if source["id"] == "moe_985" else "211"
            if code not in decoded or "工程" not in decoded or "学校" not in decoded:
                raise ValueError("Official source response lacks the expected project evidence")
        (output / source["filename"]).write_bytes(raw)
        captured.append({**source, "resolved_url": final_url, "bytes": len(raw), "sha256": digest(raw),
                         "response_headers": headers, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                         "license_boundary": "Source rights retained; private research capture, no relicensing of source documents."})
    resources = {
        "schema_version": 1, "policy_version": "context-reading-v1", "provenance": captured,
        "slang": [{"id": "u1s1", "surface": "u1s1", "case_sensitive": False,
                   "spoken_form": "有一说一", "evidence": ["u1s1_research"],
                   "reading_basis": "Source-documented meaning; expansion is the authorized project reading policy."}],
        "school_codes": [
            {"id": "project_985", "surface": "985", "spoken_form": "九八五", "evidence": ["moe_985"],
             "reading_basis": "Project convention: school-project identifier digits; official source establishes identity only."},
            {"id": "project_211", "surface": "211", "spoken_form": "二幺幺", "evidence": ["moe_211"],
             "reading_basis": "User-confirmed convention; 二幺幺 is not claimed as the sole official pronunciation."},
        ],
        "context": {"window_codepoints": 64,
                    "education_terms": ["大学", "高校", "院校", "学校", "学历", "本科", "硕士", "博士",
                                        "毕业", "考研", "高考", "招生", "名校", "学位", "双一流"],
                    "project_suffixes": ["工程"],
                    "boundary": "Finite local paragraph window; explicit number units and quantity/date/identifier syntax take precedence."},
        "evidence_boundary": "Independent source facts and user reading policy only. No evaluation texts, gold labels, predictions, training or model-benefit claim.",
    }
    write_json(output / "resources.json", resources)
    shutil.copyfile(Path(__file__), output / "builder.py")
    files = [{"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}
             for path in sorted(output.iterdir()) if path.is_file()]
    manifest = {"schema_version": 1, "status": "complete", "resources_sha256": digest((output / "resources.json").read_bytes()),
                "source_count": len(captured), "created_at": datetime.now(timezone.utc).isoformat(), "files": files}
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
