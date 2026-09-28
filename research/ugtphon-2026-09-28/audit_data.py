"""Bounded public-data audit; emits aggregates only, not social-media text."""
import collections as C
import concurrent.futures as F
import hashlib
import json
import re
import unicodedata as U
import urllib.request as R
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHA = "da2b3ee84e15e98ba48b388f7ee291ba4c3579b2"
API = "https://api.github.com/repos/naver-ai/UGTPHON"
HEADERS = {"User-Agent": "UGTPhon-bounded-research-audit"}
PATHS = [f"{lang}/{split}.json" for lang in ("en", "vi") for split in ("train", "dev", "test")]

def get(url, cap=2_000_000):
    with R.urlopen(R.Request(url, headers=HEADERS), timeout=90) as response:
        raw = response.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("Response exceeds explicit cap: " + url)
    return raw

def norm(value):
    return " ".join(U.normalize("NFC", str(value)).split())

def fold(value):
    return norm(value).casefold()

def altcanon(value):
    # Sensitivity metric only: honor numeric commas; strip peripheral punctuation
    # from other comma-delimited forms following the dataset README guidance.
    value = norm(value)
    if re.search(r"\d,\d", value):
        return {value}
    out = set()
    for part in value.split(","):
        part = part.strip()
        while part and U.category(part[0]).startswith("P"):
            part = part[1:].lstrip()
        while part and U.category(part[-1]).startswith("P"):
            part = part[:-1].rstrip()
        if part:
            out.add(norm(part))
    return out or {value}

def overlap(a, b):
    ca, cb = C.Counter(a), C.Counter(b)
    inter = ca.keys() & cb.keys()
    return {"unique_overlap": len(inter), "left_rows_in_overlap": sum(ca[k] for k in inter),
            "right_rows_in_overlap": sum(cb[k] for k in inter)}

def token_rows(rows):
    return [(row, i, tok) for row in rows for i, tok in enumerate(row["tokens"]) if tok.get("is_nc")]

def signature(row):
    # This proxy is NOT a released augmentation family ID.
    return norm(" ".join("<NC:" + str(t.get("nc_type")) + ">" if t.get("is_nc") else str(t.get("canonical", "")) for t in row["tokens"]))

def analyze(rows):
    out = {"splits": {}}
    for split, data in rows.items():
        out["splits"][split] = {
            "rows": len(data), "source_counts": dict(C.Counter(x.get("source") for x in data)),
            "all_tokens": sum(len(x["tokens"]) for x in data),
            "nc_tokens": sum(t.get("is_nc", False) for x in data for t in x["tokens"]),
            "nc_categories": dict(C.Counter(t.get("nc_type") for x in data for t in x["tokens"] if t.get("is_nc"))),
            "raw_text_nfc_changed_rows": sum(x["text"] != U.normalize("NFC", x["text"]) for x in data),
            "within_split_duplicate_raw_rows": len(data) - len({norm(x["text"]) for x in data}),
            "within_split_duplicate_canonical_rows": len(data) - len({norm(x["canonical"]) for x in data}),
            "row_keys": sorted({k for x in data for k in x}),
            "token_keys": sorted({k for x in data for t in x["tokens"] for k in t}),
        }
    out["cross_split"] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        pair = {}
        for name, fn in (("raw_nfc_whitespace", lambda x: norm(x["text"])),
                         ("raw_nfc_whitespace_casefold", lambda x: fold(x["text"])),
                         ("canonical_nfc_whitespace", lambda x: norm(x["canonical"])),
                         ("canonical_nfc_whitespace_casefold", lambda x: fold(x["canonical"])),
                         ("nc_masked_canonical_skeleton_proxy", signature)):
            pair[name] = overlap([fn(x) for x in rows[left]], [fn(x) for x in rows[right]])
        lkeys = {norm(x["text"]) for x in rows[left]}
        pair["right_duplicate_rows_by_source"] = dict(C.Counter(x.get("source") for x in rows[right] if norm(x["text"]) in lkeys))
        # Same full sentence, same marked surface, different canonical annotation.
        labels = C.defaultdict(set)
        for r, i, t in token_rows(rows[left]):
            labels[(norm(r["text"]), i, norm(t["text"]))].add(norm(t["canonical"]))
        pair["same_context_target_conflicting_canonical_right_tokens"] = sum(
            bool(labels.get((norm(r["text"]), i, norm(t["text"])))) and norm(t["canonical"]) not in labels[(norm(r["text"]), i, norm(t["text"]))]
            for r, i, t in token_rows(rows[right]))
        out["cross_split"][left + "_" + right] = pair

    out["lexical_normalization"] = {}
    for keyname, keyfn in (("nfc_whitespace_exact_surface", norm), ("nfc_whitespace_casefold_surface", fold)):
        counts = C.defaultdict(C.Counter)
        for r, i, t in token_rows(rows["train"]):
            counts[keyfn(t["text"])][norm(t["canonical"])] += 1
        major = {k: sorted(v.items(), key=lambda x: (-x[1], x[0]))[0][0] for k, v in counts.items()}
        tests = token_rows(rows["test"])
        hit = [(r, i, t) for r, i, t in tests if keyfn(t["text"]) in counts]
        ambiguous_hit = [(r, i, t) for r, i, t in hit if len(counts[keyfn(t["text"])]) > 1]
        total = len(tests)
        correct = lambda t: norm(t["canonical"]) == major[keyfn(t["text"])]
        altcorrect = lambda t: bool(altcanon(t["canonical"]) & altcanon(major[keyfn(t["text"])]))
        testlabels = C.defaultdict(C.Counter)
        contexts = C.defaultdict(set)
        test_canonical_folded = C.defaultdict(set)
        ipa_counts = C.defaultdict(C.Counter)
        for r, i, t in token_rows(rows["train"]):
            if norm(t.get("phoneme", "")):
                ipa_counts[keyfn(t["text"])][norm(t["phoneme"])] += 1
        ipa_major = {k: sorted(v.items(), key=lambda x: (-x[1], x[0]))[0][0] for k, v in ipa_counts.items()}
        ipa_test = [(r, i, t) for r, i, t in tests if norm(t.get("phoneme", ""))]
        ipa_hits = [(r, i, t) for r, i, t in ipa_test if keyfn(t["text"]) in ipa_major]
        ipa_em = sum(norm(t["phoneme"]) == ipa_major[keyfn(t["text"])] for r, i, t in ipa_hits)
        ipa_variant_em = sum(bool({norm(x) for x in t["phoneme"].split(",")} & {norm(x) for x in ipa_major[keyfn(t["text"])].split(",")}) for r, i, t in ipa_hits)
        for r, i, t in tests:
            testlabels[keyfn(t["text"])][norm(t["canonical"])] += 1
            contexts[keyfn(t["text"])].add(norm(r["text"]))
            test_canonical_folded[keyfn(t["text"])].add(fold(t["canonical"]))
        contextual_test_surfaces = {k for k, v in testlabels.items() if len(v) > 1 and len(contexts[k]) > 1}
        ambiguous_in_both = {k for k in contextual_test_surfaces if k in counts and len(counts[k]) > 1}
        result = {
            "train_distinct_nc_surfaces": len(counts), "train_multilabel_nc_surfaces": sum(len(v) > 1 for v in counts.values()),
            "test_nc_tokens": total, "test_hit_tokens": len(hit), "test_hit_rate": len(hit) / total,
            "test_unseen_surface_tokens": total - len(hit),
            "train_majority_strict_em_correct_on_hits": sum(correct(t) for r, i, t in hit),
            "train_majority_strict_em_rate_on_hits": sum(correct(t) for r, i, t in hit) / len(hit),
            "train_majority_strict_em_rate_all_test_unseen_counted_wrong": sum(correct(t) for r, i, t in hit) / total,
            "train_majority_variant_tolerant_correct_on_hits": sum(altcorrect(t) for r, i, t in hit),
            "test_hit_gold_canonical_seen_in_any_train_variant": sum(norm(t["canonical"]) in counts[keyfn(t["text"])] for r, i, t in hit),
            "test_tokens_with_train_ambiguous_surface": len(ambiguous_hit),
            "test_ambiguous_hit_strict_em_correct": sum(correct(t) for r, i, t in ambiguous_hit),
            "test_multilabel_surface_count_with_distinct_contexts": len(contextual_test_surfaces),
            "test_multilabel_surface_count_after_canonical_casefold": sum(len(v) > 1 for v in test_canonical_folded.values()),
            "test_tokens_on_multilabel_surface_with_distinct_contexts": sum(sum(testlabels[k].values()) for k in contextual_test_surfaces),
            "surfaces_multilabel_in_train_and_test_with_distinct_test_contexts": len(ambiguous_in_both),
            "test_optimal_context_free_dictionary_error_lower_bound_tokens": sum(sum(v.values()) - max(v.values()) for v in testlabels.values()),
            "source_breakdown": {}, "category_breakdown": {},
            "train_majority_ipa": {"train_surface_keys_with_ipa": len(ipa_counts), "test_nc_tokens_with_nonempty_ipa": len(ipa_test),
                "test_hit_tokens": len(ipa_hits), "test_hit_rate": len(ipa_hits) / len(ipa_test),
                "exact_ipa_match_count_on_hits": ipa_em, "exact_ipa_match_rate_on_hits": ipa_em / len(ipa_hits),
                "exact_ipa_match_rate_all_nonempty_test_missing_wrong": ipa_em / len(ipa_test),
                "variant_tolerant_ipa_match_count_on_hits": ipa_variant_em,
                "variant_tolerant_ipa_match_rate_all_nonempty_test_missing_wrong": ipa_variant_em / len(ipa_test),
                "note": "String-level IPA exact match; NOT paper PER. Variant tolerance splits comma alternatives only."},
        }
        for field, dest in (("source", "source_breakdown"), ("nc_type", "category_breakdown")):
            vals = sorted({r.get(field) if field == "source" else t.get(field) for r, i, t in tests}, key=str)
            for val in vals:
                subset = [(r, i, t) for r, i, t in tests if (r.get(field) if field == "source" else t.get(field)) == val]
                hits = [(r, i, t) for r, i, t in subset if keyfn(t["text"]) in counts]
                result[dest][str(val)] = {"nc_tokens": len(subset), "hit_tokens": len(hits), "majority_em_correct": sum(correct(t) for r, i, t in hits)}
        # Whitelist only generic, nonidentifying lexical forms; no sentences/IDs/usernames.
        examples = {}
        whitelist = {"ig", "im", "id", "ur", "u", "lol", "smh", "bc", "ppl", "k", "ko", "dc", "đc", "kh", "j", "m", "t", "vs", "mk", "b", "ma", "a", "c"}
        for k in sorted(contextual_test_surfaces & whitelist):
            examples[k] = {"train_canonical_counts": dict(counts.get(k, {})), "test_canonical_counts": dict(testlabels[k]), "distinct_test_context_count": len(contexts[k])}
        result["generic_lexeme_examples_no_sentences"] = examples
        out["lexical_normalization"][keyname] = result

    # Augmentation relations: canonical equality/skeleton are measured proxies only.
    out["augmentation"] = {}
    for split in ("train", "dev", "test"):
        aug = [r for r in rows[split] if r.get("source") == "augmented"]
        other = [r for sp, rs in rows.items() if sp != split for r in rs]
        out["augmentation"][split] = {"rows": len(aug),
            "canonical_sentence_matches_other_splits": overlap([norm(r["canonical"]) for r in aug], [norm(r["canonical"]) for r in other]),
            "masked_skeleton_matches_other_splits_proxy": overlap([signature(r) for r in aug], [signature(r) for r in other])}
    return out

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    current = json.loads(get(API + "/commits/main"))
    tree = json.loads(get(API + f"/git/trees/{SHA}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("Cannot verify full tree")
    files = {x["path"]: x for x in tree["tree"] if x["path"] in PATHS}
    if len(files) != 6 or sum(x["size"] for x in files.values()) > 60_000_000:
        raise ValueError("Unexpected corpus size or missing files")
    manifest = {"checked_current_main": current["sha"], "pinned_commit": SHA, "total_declared_bytes": sum(x["size"] for x in files.values()), "files": []}
    def fetch(path):
        info = files[path]
        target = ROOT / "data" / path
        url = f"https://raw.githubusercontent.com/naver-ai/UGTPHON/{SHA}/{path}"
        raw = target.read_bytes() if target.exists() else get(url, info["size"])
        gitsha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if len(raw) != info["size"] or gitsha != info["sha"]:
            raise ValueError("Content does not match pinned tree: " + path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return path, json.loads(raw), {"path": path, "bytes": len(raw), "git_blob_sha1": gitsha, "sha256": hashlib.sha256(raw).hexdigest(), "url": url}
    datasets = {"en": {}, "vi": {}}
    with F.ThreadPoolExecutor(max_workers=3) as pool:
        for path, rows, record in pool.map(fetch, PATHS):
            lang, split = path.removesuffix(".json").split("/")
            datasets[lang][split] = rows
            manifest["files"].append(record)
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"method": {"content_keys": "NFC + collapse whitespace; primary case sensitive; supplemental casefold", "majority_tie_break": "largest train count, then lexicographic canonical", "em": "normalization exact match, NOT phoneme error rate", "no_inference_from_ids": True}, "languages": {lang: analyze(rows) for lang, rows in datasets.items()}}
    (ROOT / "aggregate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {}
    for lang, a in report["languages"].items():
        lex = a["lexical_normalization"]["nfc_whitespace_exact_surface"]
        compact[lang] = {"splits": {k: {x: v[x] for x in ("rows", "nc_tokens", "source_counts", "within_split_duplicate_raw_rows")} for k,v in a["splits"].items()}, "cross_split": a["cross_split"], "lexical": {k:v for k,v in lex.items() if k not in ("source_breakdown", "category_breakdown")}, "augmentation": a["augmentation"]}
    print(json.dumps({"bytes": manifest["total_declared_bytes"], "results": compact}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
