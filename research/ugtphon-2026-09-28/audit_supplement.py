"""Decompose lookup errors and ambiguity without outputting source sentences."""
import collections as C
import json
import unicodedata as U
from audit_data import ROOT, norm, fold, token_rows, altcanon

out = {}
for lang in ("en", "vi"):
    rows = {s: json.loads((ROOT / "data" / lang / (s + ".json")).read_text(encoding="utf-8")) for s in ("train", "dev", "test")}
    train, test = token_rows(rows["train"]), token_rows(rows["test"])
    raw_keys = {t["text"] for r, i, t in train}
    canons, ipa, paired = C.defaultdict(C.Counter), C.defaultdict(C.Counter), C.defaultdict(C.Counter)
    for r, i, t in train:
        k, c, p = norm(t["text"]), norm(t["canonical"]), norm(t.get("phoneme", ""))
        canons[k][c] += 1
        if p:
            ipa[k][p] += 1
            paired[k, c][p] += 1
    majority = lambda counts: {k: sorted(v, key=lambda x: (-v[x], x))[0] for k,v in counts.items()}
    cm, pm = majority(canons), majority(ipa)
    breakdown = C.Counter()
    testlabels = C.defaultdict(C.Counter)
    for r, i, t in test:
        k, c, p = norm(t["text"]), norm(t["canonical"]), norm(t.get("phoneme", ""))
        testlabels[k][c] += 1
        if k not in cm:
            continue
        ce, pe = c == cm[k], p == pm.get(k)
        breakdown[f"canonical_{'correct' if ce else 'wrong'}_ipa_{'correct' if pe else 'wrong'}"] += 1
        if ce and not pe:
            breakdown["canonical_correct_ipa_wrong_gold_ipa_seen_for_same_train_surface_canonical" if p in paired[k,c] else "canonical_correct_ipa_wrong_gold_ipa_NOT_seen_for_same_train_surface_canonical"] += 1
    # Connected accepted-form sets collapse only punctuation/case/explicit alternatives.
    # Remaining divergence is lexical annotation divergence, not proved semantic ambiguity.
    groups = {}
    lowerbound_fold = 0
    for k, v in testlabels.items():
        cv = C.Counter()
        components = []
        for label, count in v.items():
            key = fold(label)
            cv[key] += count
            aset = {fold(x) for x in altcanon(label)}
            linked = [g for g in components if g & aset]
            for g in linked:
                aset |= g
                components.remove(g)
            components.append(aset)
        groups[k] = len(components)
        lowerbound_fold += sum(cv.values()) - max(cv.values())
    generic = {}
    whitelist = {"you", "your", "you're", "không", "tôi", "tao", "được", "mình", "mày", "cậu", "chị"}
    for k in ("ur", "u", "k", "ko", "t", "m", "c", "đc", "dc"):
        if k not in canons:
            continue
        generic[k] = {c: {"train_ipa_counts": dict(paired[k,c]), "test_ipa_counts": dict(C.Counter(norm(t.get("phoneme", "")) for r,i,t in test if norm(t["text"]) == k and norm(t["canonical"]) == c))} for c in canons[k] if c in whitelist}
    # Inference never reads the gold NC flag; that flag is used only for reporting.
    ungated = {"canonical": C.Counter(), "noncanonical": C.Counter()}
    wrong_mappings = C.Counter()
    relaxed_harm = C.Counter()
    def flex(s):
        return " ".join("".join(c for c in fold(s) if not U.category(c).startswith("P")).split())
    for r in rows["test"]:
        for t in r["tokens"]:
            surface, gold = norm(t["text"]), norm(t["canonical"])
            predicted = cm.get(surface, surface)
            metrics = ungated["noncanonical" if t.get("is_nc") else "canonical"]
            metrics["tokens"] += 1
            metrics["lookup_hit"] += surface in cm
            metrics["rewritten"] += predicted != surface
            metrics["normalization_em_correct"] += predicted == gold
            metrics["leave_as_is_em_correct"] += surface == gold
            metrics["rewritten_but_wrong_vs_gold"] += predicted != surface and predicted != gold
            metrics["introduced_error_from_correct_leave_as_is"] += surface == gold and predicted != gold
            metrics["fixed_leave_as_is_error"] += surface != gold and predicted == gold
            if not t.get("is_nc") and predicted != surface and predicted != gold:
                wrong_mappings[surface, predicted, gold] += 1
                relaxed_harm["wrong_rewrites_after_casefold"] += fold(predicted) != fold(gold)
                relaxed_harm["wrong_rewrites_after_casefold_remove_punctuation"] += flex(predicted) != flex(gold)
                relaxed_harm["introduced_errors_after_casefold_remove_punctuation"] += flex(surface) == flex(gold) and flex(predicted) != flex(gold)
    ungated = {k: {**dict(v), "em_rate": v["normalization_em_correct"] / v["tokens"],
                           "wrong_rewrite_rate_all_tokens": v["rewritten_but_wrong_vs_gold"] / v["tokens"]} for k,v in ungated.items()}
    out[lang] = {"raw_surface_no_normalization_train_keys": len(raw_keys),
                 "raw_surface_no_normalization_test_hits": sum(t["text"] in raw_keys for r,i,t in test),
                 "lookup_error_breakdown": dict(breakdown),
                 "test_multilabel_surfaces_after_case_punctuation_alternative_collapse": sum(n > 1 for n in groups.values()),
                 "test_context_free_oracle_lower_bound_errors_after_canonical_casefold": lowerbound_fold,
                 "train_surface_canonical_pairs_with_multiple_ipa_strings": sum(len(v)>1 for v in paired.values()),
                 "train_surface_canonical_pairs_total": len(paired),
                 "ungated_lookup_all_tokens_miss_identity_no_gold_routing": ungated,
                 "canonical_wrong_rewrite_sensitivity": dict(relaxed_harm),
                 "top_10_wrong_canonical_rewrite_word_triples_count_at_least_10_no_sentences": [dict(surface=k[0], prediction=k[1], gold=k[2], count=v) for k,v in wrong_mappings.most_common(10) if v>=10],
                 "generic_lexemes_ipa_variants_no_sentences": generic}
(ROOT / "supplement.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
