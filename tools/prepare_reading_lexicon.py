"""Build a small, sourced reading-policy resource without evaluation inputs.

The vocabulary is intentionally finite. Official spelling/name evidence is not
presented as pronunciation evidence. This script performs no network access,
model inference, corpus inspection, or paid work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from collections import Counter
from pathlib import Path


RESOURCE_ID = "reading-lexicon-v2-2026-10-01"
RESEARCH_DATE = "2026-10-01"
DEFAULT_OUTPUT = Path("E:/CodexRuntime/ReadCue/data") / RESOURCE_ID

# Public primary pages were read on RESEARCH_DATE. These are factual metadata
# and original summaries, not downloaded or redistributed page snapshots.
SOURCE_ROWS = [
    ("user_policy", None, "Explicit user reading policy", "user_policy",
     "The user specified up as a word, YYDS and JDK as letters, and tomcat as a word."),
    ("oracle_java", "https://docs.oracle.com/javase/7/docs/technotes/guides/",
     "Oracle Java Platform Overview", "official_name",
     "Oracle names Java, Java SE Development Kit (JDK), and JRE; this establishes names, not their spoken realization."),
    ("apache_tomcat", "https://tomcat.apache.org/", "Apache Tomcat Welcome", "official_name",
     "The Apache project identifies its software as Tomcat and lists numbered release families."),
    ("khronos_opengl", "https://registry.khronos.org/OpenGL/index_gl.php",
     "Khronos OpenGL Registry", "official_name",
     "The registry identifies OpenGL and uses the token API for its specifications; it does not prescribe letter-name pronunciation."),
    ("whatwg_html", "https://html.spec.whatwg.org/index.html", "WHATWG HTML Standard", "official_name",
     "The standard identifies HTML as the web markup language; the page establishes spelling, not pronunciation."),
    ("w3c_css", "https://www.w3.org/Style/CSS/Overview.en.html", "W3C Cascading Style Sheets", "official_name",
     "W3C identifies Cascading Style Sheets with the abbreviation CSS; the page does not prescribe spoken letter names."),
    ("rfc_http", "https://www.rfc-editor.org/rfc/rfc9110.html", "RFC 9110 HTTP Semantics", "official_name",
     "The standard identifies HTTP and defines the http and https URI schemes; no pronunciation rule is claimed."),
    ("whatwg_url", "https://url.spec.whatwg.org/", "WHATWG URL Standard", "official_name",
     "The standard establishes the URL token; it does not select a pronunciation."),
    ("rfc_uri", "https://www.rfc-editor.org/rfc/rfc3986.html", "RFC 3986 URI Generic Syntax", "official_name",
     "The RFC names Uniform Resource Identifier (URI); its name does not establish spoken letter names."),
    ("rfc_dns", "https://www.rfc-editor.org/rfc/rfc1034.html", "RFC 1034 Domain Names Concepts and Facilities", "spelling",
     "The RFC uses DNS throughout its description of the domain name system; this supports spelling only."),
    ("intel_cpu", "https://www.intel.com/content/www/us/en/products/docs/processors/cpu-vs-gpu.html",
     "Intel CPU versus GPU", "official_name",
     "Intel identifies central processing units as CPUs and graphics processing units as GPUs; no pronunciation is prescribed."),
    ("nvidia_gpu", "https://www.nvidia.com/content/Control-Panel-Help/vLatest/en-us/mergedProjects/nvcpl/glossary.htm",
     "NVIDIA Control Panel Glossary", "official_name",
     "The manufacturer's glossary identifies graphics processing units with GPU; this is naming evidence only."),
    ("android_sdk", "https://developer.android.com/tools", "Android Command-line Tools", "spelling",
     "Android's documentation identifies its SDK tool packages; this supports the SDK spelling, not an official pronunciation."),
    ("python_faq", "https://docs.python.org/3/faq/general.html", "General Python FAQ", "official_name",
     "The Python project explains its name and A.B.C or A.B version components; Chinese readout is a project convention."),
    ("docker_overview", "https://docs.docker.com/get-started/docker-overview/", "Docker Overview", "official_name",
     "Docker's documentation establishes the Docker product name and spelling, without prescribing pronunciation."),
    ("apple_iphone", "https://www.apple.com/iphone-16/specs/", "Apple iPhone 16 Technical Specifications", "official_name",
     "Apple identifies the product as iPhone 16; this supports the word stem and numeric model structure, not Chinese readout."),
    ("nvidia_rtx", "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/",
     "NVIDIA GeForce RTX 4090", "official_name",
     "NVIDIA uses the RTX stem in a numeric product name; letter spelling and integer readout here are project conventions."),
    ("mysql_manual", "https://dev.mysql.com/doc/refman/8.4/en/what-is-mysql.html", "MySQL Reference Manual: What Is MySQL?", "pronunciation",
     "The manual gives My Ess Que Ell as the official pronunciation, while accepting alternatives; it also identifies SQL by name."),
    ("json_project", "https://www.json.org/json-en.html", "JSON Project", "official_name",
     "The project page identifies JavaScript Object Notation as JSON; this page is not treated as unique-pronunciation evidence."),
    ("ibm_ai", "https://www.ibm.com/cn-zh/think/topics/artificial-intelligence",
     "IBM: What Is Artificial Intelligence (AI)?", "spelling",
     "IBM's own explanation explicitly uses AI for artificial intelligence. This supports the term and spelling, not an official letter-name pronunciation."),
    ("google_llm", "https://cloud.google.com/ai/llms?hl=en", "Google Cloud: Large Language Models", "spelling",
     "Google Cloud identifies large language model with LLM. The page provides naming and spelling evidence, not a spoken pronunciation rule."),
    ("openai_chatgpt", "https://developers.openai.com/chatgpt", "OpenAI Developers: Build for ChatGPT", "official_name",
     "OpenAI's own developer page uses the product name ChatGPT. Splitting it into the word Chat plus the letters GPT is this project's convention, not an official pronunciation claim."),
]


def sources() -> list[dict]:
    return [
        {
            "id": source_id,
            "url": url,
            "title": title,
            "source_kind": "explicit_user_policy" if url is None else "public_primary",
            "evidence_type": evidence_type,
            "claim": claim,
            "accessed_at": RESEARCH_DATE,
            "evidence_scope": "Task instruction" if url is None else "Page text inspected; no audio pronunciation verification",
        }
        for source_id, url, title, evidence_type, claim in SOURCE_ROWS
    ]


def evidence(source_id: str, evidence_type: str | None = None) -> dict:
    source = next(item for item in sources() if item["id"] == source_id)
    result = {key: source[key] for key in ("url", "title", "evidence_type", "claim", "accessed_at")}
    result["source_id"] = source_id
    if evidence_type:
        result["evidence_type"] = evidence_type
    if source_id == "mysql_manual" and evidence_type == "official_name":
        result["claim"] = "The manual identifies SQL as Structured Query Language. Its MySQL pronunciation statement does not prescribe a unique standalone SQL pronunciation."
    if source_id == "khronos_opengl" and evidence_type == "spelling":
        result["claim"] = "The official registry uses the token API. This supports spelling only, not pronunciation."
    return result


def make_entry(
    term: str,
    kind: str,
    action: str,
    source_ids: list[str],
    *,
    reading_basis: str = "project_convention",
    components: list[dict] | None = None,
    base_action: str | None = None,
    notes: list[str] | None = None,
) -> dict:
    if action == "spell_letters":
        spoken_form = " ".join(term.upper())
    elif action == "compose":
        spoken_form = " ".join(
            " ".join(part["text"].upper()) if part["action"] == "spell_letters" else part["text"]
            for part in components or []
        )
    else:
        spoken_form = term
    item = {
        "id": term.lower(),
        "surfaces": [{"text": term, "case_sensitive": False}],
        "kind": kind,
        "action": action,
        "spoken_form": spoken_form,
        "reading_basis": reading_basis,
        "evidence": [evidence(source_id) for source_id in source_ids],
        "ambiguity_notes": notes or [],
        "case_policy_basis": "Project convention: recognize ASCII case variants; preserve matched word casing.",
        "requires_review": action == "preserve_review",
    }
    if components:
        item["components"] = components
    if base_action:
        item.update(
            base_action=base_action,
            version_policy="integer_components",
            version_reading_basis="project_convention",
        )
    return item


def entries() -> list[dict]:
    result = [
        make_entry("up", "word", "read_word", ["user_policy"], reading_basis="explicit_user_policy"),
        make_entry("YYDS", "initialism", "spell_letters", ["user_policy"], reading_basis="explicit_user_policy"),
        make_entry("JDK", "version_stem", "spell_letters", ["oracle_java", "user_policy"], reading_basis="explicit_user_policy", base_action="letters"),
        make_entry("Tomcat", "version_stem", "read_word", ["apache_tomcat", "user_policy"], reading_basis="explicit_user_policy", base_action="word"),
        make_entry("API", "initialism", "spell_letters", ["khronos_opengl"]),
        make_entry("HTML", "initialism", "spell_letters", ["whatwg_html"]),
        make_entry("CSS", "initialism", "spell_letters", ["w3c_css"]),
        make_entry("HTTP", "initialism", "spell_letters", ["rfc_http"]),
        make_entry("HTTPS", "initialism", "spell_letters", ["rfc_http"]),
        make_entry("URL", "initialism", "spell_letters", ["whatwg_url"]),
        make_entry("URI", "initialism", "spell_letters", ["rfc_uri"]),
        make_entry("DNS", "initialism", "spell_letters", ["rfc_dns"]),
        make_entry("CPU", "initialism", "spell_letters", ["intel_cpu"]),
        make_entry("GPU", "initialism", "spell_letters", ["nvidia_gpu"]),
        make_entry("SDK", "initialism", "spell_letters", ["android_sdk"]),
        make_entry("Python", "version_stem", "read_word", ["python_faq"], base_action="word"),
        make_entry("Java", "version_stem", "read_word", ["oracle_java"], base_action="word"),
        make_entry("Docker", "word", "read_word", ["docker_overview"]),
        make_entry("iPhone", "version_stem", "read_word", ["apple_iphone"], base_action="word"),
        make_entry("RTX", "version_stem", "spell_letters", ["nvidia_rtx"], base_action="letters", notes=["Chinese integer model-number readout is a project choice; the source does not select between colloquial model-name pronunciations."]),
        make_entry("MySQL", "compound", "compose", ["mysql_manual"], reading_basis="official_pronunciation", components=[{"text": "My", "action": "read_word"}, {"text": "SQL", "action": "spell_letters"}], notes=["The manual explicitly accepts alternative pronunciations. This project selects its stated official form; standalone SQL remains unresolved."]),
        make_entry("OpenGL", "compound", "compose", ["khronos_opengl"], components=[{"text": "Open", "action": "read_word"}, {"text": "GL", "action": "spell_letters"}]),
        make_entry("JSON", "ambiguous", "preserve_review", ["json_project"], notes=["This project conservatively leaves JSON reading unresolved. Preserve the complete matched spelling and emit a review warning; this is not an explicit user pronunciation instruction."]),
        make_entry("SQL", "ambiguous", "preserve_review", ["mysql_manual"], notes=["This project conservatively leaves standalone SQL reading unresolved. The MySQL compound's documented pronunciation is not a rule for every SQL occurrence; this is not an explicit user pronunciation instruction."]),
        make_entry("AI", "initialism", "spell_letters", ["ibm_ai"], notes=["Selected for the authorized AI-technology-comment domain. The source establishes spelling; letter-name readout is a project convention."]),
        make_entry("LLM", "initialism", "spell_letters", ["google_llm"], notes=["Selected for the authorized AI-technology-comment domain. The source establishes spelling; letter-name readout is a project convention."]),
        make_entry("ChatGPT", "compound", "compose", ["openai_chatgpt"], components=[{"text": "Chat", "action": "read_word"}, {"text": "GPT", "action": "spell_letters"}], notes=["Selected for the authorized AI-technology-comment domain. Preserve the matched Chat word segment's case and spell GPT; this is not claimed as an official OpenAI pronunciation."]),
    ]
    for item in result:
        if item["id"] == "api":
            item["evidence"][0] = evidence("khronos_opengl", "spelling")
        elif item["id"] == "sql":
            item["evidence"][0] = evidence("mysql_manual", "official_name")
    return result


def validate(items: list[dict]) -> None:
    if len(items) != 27:
        raise ValueError("The v2 resource must contain exactly 27 reviewed entries")
    known_sources = {item["id"] for item in sources()}
    seen_ids: set[str] = set()
    seen_surfaces: set[str] = set()
    for item in items:
        if item["id"] in seen_ids:
            raise ValueError("Duplicate entry id")
        seen_ids.add(item["id"])
        if not item["evidence"] or not item["spoken_form"]:
            raise ValueError("Every entry requires evidence and directly usable text")
        for proof in item["evidence"]:
            if proof["source_id"] not in known_sources:
                raise ValueError("Unknown source")
            if proof["url"] is None and proof["evidence_type"] != "user_policy":
                raise ValueError("Only explicit user policy may omit a public URL")
        for surface in item["surfaces"]:
            term = surface["text"]
            key = term.casefold()
            if key in seen_surfaces or not re.fullmatch(r"[A-Za-z]+", term):
                raise ValueError("Duplicate or unsupported surface")
            seen_surfaces.add(key)
        if item["kind"] == "version_stem":
            if item["base_action"] not in {"word", "letters"} or item["version_policy"] != "integer_components":
                raise ValueError("Invalid version contract")
        if item["kind"] == "compound":
            joined = "".join(part["text"] for part in item["components"])
            if joined.casefold() != item["surfaces"][0]["text"].casefold():
                raise ValueError("Compound components do not reconstruct the surface")
            if any(part["action"] not in {"read_word", "spell_letters"} for part in item["components"]):
                raise ValueError("Unsupported compound action")
        if item["kind"] == "ambiguous" and item["action"] != "preserve_review":
            raise ValueError("Ambiguous terms must remain unchanged")
        if item["reading_basis"] == "official_pronunciation" and not any(proof["evidence_type"] == "pronunciation" for proof in item["evidence"]):
            raise ValueError("Official pronunciation requires direct evidence")


def encode(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build() -> dict[str, bytes]:
    items = entries()
    validate(items)
    lexicon = {
        "schema_version": 1,
        "resource_id": RESOURCE_ID,
        "policy_version": "english-reading-v2",
        "research_date": RESEARCH_DATE,
        "frozen": True,
        "entry_count": len(items),
        "provenance": {
            "selection": "A finite selection of common technology names from public primary documents, plus the explicit current user reading policy.",
            "evaluation_inputs_read_for_selection": False,
            "evaluation_labels_or_predictions_read_for_selection": False,
            "prior_exposure_disclosure": "The author previously authored and audited a separate 24-case comment-policy challenge. None of that material was reread or mined for this resource. This author is not a blind-evaluation author for the present round.",
            "primary_document_names_are_not_pronunciation_proof": True,
            "human_reviewed": False,
        },
        "revision_from_v1": {
            "previous_resource_id": "reading-lexicon-v1-2026-10-01",
            "previous_lexicon_sha256": "b2cc2bc01c3407d309585aaf5b220fd3c6eb91b9a7f6e4f165d2d4bf633d7913",
            "previous_builder_snapshot_sha256": "9695ac371e09fbc3663eea9ee8a239091497dfc0873d7381c5c3fe4c0d5e4b4f",
            "reason": "Pre-inference source review corrected JSON/SQL provenance from explicit_user_policy to project_convention and added only AI, LLM, and ChatGPT for the authorized task domain.",
            "new_round_evaluation_references_or_predictions_read": False,
            "evaluation_labels_modified": False,
        },
        "matching_contract": {
            "boundary": "Exact known term with ASCII identifier boundaries; do not match substrings inside longer ASCII words or identifiers.",
            "protected_contexts": "The consumer must preserve code, arrays, URLs, and other protected literal text before term matching.",
            "word_rendering": "Keep the original matched spelling and case. spoken_form gives the canonical-surface output, not permission to overwrite case.",
            "letter_rendering": "Uppercase ASCII letters separated by single ASCII spaces.",
            "compound_rendering": "Match the complete surface first; slice original text using component lengths, preserve word component case, spell letter components, and join components with single ASCII spaces.",
            "version_rendering": "For selected version_stem entries, render each integer component in Chinese; preserve leading-zero components by reading each digit; join multiple components with 点. Never parse versions as floating-point numbers. This is a project convention, not an official vendor pronunciation claim.",
            "unknown": "Preserve. Never infer spelling-out merely from uppercase text.",
            "ambiguous": "Preserve the full original matched term and emit a review warning.",
        },
        "entries": items,
    }
    source_document = {
        "schema_version": 1,
        "resource_id": RESOURCE_ID,
        "research_date": RESEARCH_DATE,
        "sources": sources(),
        "retention": "Metadata and original summaries only; no raw webpages, audio, dictionaries, or licensed datasets are bundled.",
        "license_boundary": "Repository code licensing does not replace source-site or trademark terms. No permission to redistribute source pages or audio is claimed.",
    }
    readme = {
        "resource_id": RESOURCE_ID,
        "status": "frozen_for_experiment",
        "schema_version": 1,
        "counts_by_kind": dict(sorted(Counter(item["kind"] for item in items).items())),
        "selection_strategy": [
            "Select a small general vocabulary from public primary technology standards, project pages, and vendor naming documents.",
            "Keep explicit current user choices in a separately identified user_policy source; its missing URL is intentional and is not fabricated public evidence.",
            "Treat initialism spelling-out and most English-word handling as project conventions unless the source explicitly states pronunciation.",
            "Use MySQL's documented preferred pronunciation for its complete compound while preserving ambiguous standalone SQL and JSON.",
            "Apply the approved integer_components version convention only to the six declared product/version stems.",
            "Do not inspect evaluation cases, outputs, or reports to mine vocabulary or adjust this frozen version.",
        ],
        "scope_limits": [
            "27 entries are not a comprehensive pronunciation dictionary.",
            "Text transformations do not establish how any TTS engine will actually pronounce the output.",
            "Word handling preserves input spelling, without inventing IPA, Chinese homophones, translations, or acronym expansions.",
            "No ReadCue checkpoint, model inference result, pronunciation evaluation, paid run, or human listening approval is produced by this builder.",
            "JSON and SQL intentionally require review; MySQL alternatives remain possible even though this project selects the manual's preferred form.",
        ],
        "reproduce": "python tools/prepare_reading_lexicon.py --output-dir <new external directory>",
        "storage": "Generated resources must remain outside the repository. The builder uses only its embedded reviewed metadata and performs no downloads.",
        "freeze_rule": "Do not edit this resource after downstream output is inspected. Changes require a new version and explicit records; an existing output directory is verified byte-for-byte, never overwritten.",
        "v2_change_record": "The prior 24-entry resource and exact builder snapshot remain in the v1 external directory. This pre-inference revision corrects the JSON/SQL policy attribution and adds only the three authorized domain terms AI, LLM, and ChatGPT from public primary naming evidence. No evaluation input, label, prediction, or report was read for this revision.",
    }
    files = {"lexicon.json": encode(lexicon), "sources.json": encode(source_document), "README.json": encode(readme)}
    manifest = {
        "resource_id": RESOURCE_ID,
        "schema_version": 1,
        "frozen": True,
        "entry_count": len(items),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(files.items())},
    }
    files["MANIFEST.json"] = encode(manifest)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo_root):
        parser.error("Generated resources must be outside the repository")
    payloads = build()
    if output.exists():
        if not output.is_dir() or {item.name for item in output.iterdir()} != set(payloads):
            parser.error("Existing resource directory does not match the frozen file set")
        if any((output / name).read_bytes() != data for name, data in payloads.items()):
            parser.error("Existing frozen resources differ; refusing to overwrite")
        status = "verified_existing"
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, data in payloads.items():
            path = output / name
            with path.open("xb") as handle:
                handle.write(data)
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        status = "created_frozen"
    print(json.dumps({
        "status": status,
        "output_dir": str(output),
        "entry_count": 27,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())},
    }, indent=2))


if __name__ == "__main__":
    main()
