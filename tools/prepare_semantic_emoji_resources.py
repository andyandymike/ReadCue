"""Build a bounded noun-emoji resource from a pinned, independently saved CLDR XML."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

COMMIT = "2ef784e3a4168bc2a43cd1b5b9839b6636f5899c"
XML_SHA256 = "06b289e1b563199df2e2c51ab93ca8a3780a57a7ee82f4600828936c8323af0d"
# Explicit small object inventory, not terms extracted from evaluation labels.
SURFACES = ("💩", "🔨", "🍎", "🍋", "🐕", "🐈", "🌹", "🧱")


def build(xml_path):
    payload = Path(xml_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != XML_SHA256:
        raise ValueError("Expected pinned CLDR 47 Chinese annotations")
    names = {node.attrib["cp"]: node.text for node in ET.fromstring(payload).findall("./annotations/annotation")
             if node.attrib.get("type") == "tts"}
    return {"schema_version": 1, "policy_version": "semantic-emoji-resource-v1",
            "provenance": {"source": "Unicode CLDR 47 Chinese tts annotations", "commit": COMMIT,
                           "url": f"https://raw.githubusercontent.com/unicode-org/cldr/{COMMIT}/common/annotations/zh.xml",
                           "source_sha256": XML_SHA256, "license": "Unicode-3.0",
                           "selection": "Eight explicitly enumerated object pictographs; use only in explicit noun slots.",
                           "scope": "Literal Chinese names, not a general slang/metaphor resolver; no gold used."},
            "entries": [{"id": "unicode_" + "_".join(f"{ord(c):x}" for c in surface),
                         "surface": surface, "spoken_form": names[surface], "evidence": "cldr47_zh_tts"}
                        for surface in SURFACES]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.xml)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
