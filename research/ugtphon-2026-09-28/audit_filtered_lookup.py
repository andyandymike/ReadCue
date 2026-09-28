import collections as C
import difflib
import json
from audit_data import ROOT, norm, token_rows

out = {}
for lang in ("en", "vi"):
    train = json.loads((ROOT / "data" / lang / "train.json").read_text(encoding="utf-8"))
    test = json.loads((ROOT / "data" / lang / "test.json").read_text(encoding="utf-8"))
    canonical_keys = {norm(t["text"]) for r in train for t in r["tokens"] if not t.get("is_nc")}
    counts = C.defaultdict(C.Counter)
    for r, i, t in token_rows(train):
        counts[norm(t["text"])][norm(t["canonical"])] += 1
    major = {k: sorted(v, key=lambda x: (-v[x], x))[0] for k,v in counts.items() if k not in canonical_keys}
    groups = {"canonical": C.Counter(), "noncanonical": C.Counter()}
    for r in test:
        for t in r["tokens"]:
            k,g = norm(t["text"]),norm(t["canonical"])
            p = major.get(k,k)
            a = groups["noncanonical" if t.get("is_nc") else "canonical"]
            a["tokens"] += 1
            a["hits"] += k in major
            a["em_correct"] += p == g
            a["introduced_errors_from_correct_identity"] += k == g and p != g
            a["rewritten_wrong_vs_gold"] += p != k and p != g
    diagnostic = []
    for r,i,t in token_rows(train):
        if (norm(t["text"]),norm(t["canonical"])) not in {("có","thú"),("đi","quảng")}:
            continue
        raw = norm(r["text"]).split()
        gold = norm(r["canonical"]).split()
        tokens_raw = norm(" ".join(x["text"] for x in r["tokens"]))
        tokens_gold = norm(" ".join(x["canonical"] for x in r["tokens"]))
        raw_index = sum(len(norm(x["text"]).split()) for x in r["tokens"][:i])
        gold_index = sum(len(norm(x["canonical"]).split()) for x in r["tokens"][:i])
        aligned = []
        for block in difflib.SequenceMatcher(None, raw, gold, autojunk=False).get_matching_blocks():
            if block.a <= raw_index < block.a + block.size:
                aligned.append({"gold_index": block.b + raw_index - block.a, "same_surface_matched": gold[block.b + raw_index - block.a] == norm(t["text"]), "equal_run_length": block.size})
        diagnostic.append({"surface":t["text"],"token_canonical":t["canonical"],"source":r["source"],"raw_word_count":len(raw),"canonical_word_count":len(gold),"raw_target_index":raw_index,"token_derived_canonical_index":gold_index,"joined_raw_tokens_equals_sentence":tokens_raw==norm(r["text"]),"joined_canonical_tokens_equals_canonical_sentence":tokens_gold==norm(r["canonical"]),"sequence_matcher_identity_alignment":aligned,"surface_nc_train_counts":dict(counts[norm(t["text"])]),"surface_also_in_train_canonical":norm(t["text"]) in canonical_keys})
    out[lang]={"train_nc_keys":len(counts),"filtered_keys":len(major),"removed_keys":len(counts)-len(major),"test_all_tokens_no_gold_routing":{k:dict(v) for k,v in groups.items()},"targeted_alignment_diagnostics_no_sentences":diagnostic}
(ROOT / "filtered-lookup.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(out,ensure_ascii=False,indent=2))
