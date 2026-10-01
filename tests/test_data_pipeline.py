"""Offline contract tests for source-only sampling and reviewed evaluation exports.

Fixtures are original synthetic comments, never downloaded source text or model
outputs.  Tests exercise the real CSV importer and the existing input/scoring
contracts without running a baseline, changing the pilot, or accessing a network.
"""
import copy
import csv
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from readcue.artifacts import hash_file, read_json, read_jsonl
from readcue.baselines import validate_inputs
from readcue.data_pipeline import export_review, prepare_data


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from score_zh_pilot import read_jsonl as legacy_read_jsonl, score


CSV_FIELDS = ["bv_id", "oid", "level", "root_rpid", "rpid", "parent_rpid",
              "uid", "uname", "user_level", "ctime", "like", "message"]
REVISION = "a" * 40
SPECIAL_TEXT = "  今天预算１２元，型号 A7 保持不变。\n下一行也保留。  "


def timestamp(day):
    return str(int(datetime.fromisoformat(day).replace(
        tzinfo=timezone(timedelta(hours=8))).timestamp()))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8", newline="\n")


def digest_tree(directory):
    return {str(path.relative_to(directory)): hash_file(path)
            for path in directory.rglob("*") if path.is_file()}


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_root = self.root / "sources"
        self.source_root.mkdir()
        self.original_by_text = {}
        self.candidate_texts = set()
        self.control_texts = set()
        self.outside_texts = set()
        files = []
        for index, (name, letter) in enumerate(zip("甲乙丙丁戊己庚", "HJKLMNP")):
            video = "BV1ABcdefg" + letter + "2"
            rows = []
            for position, season in enumerate("春夏秋冬晴"):
                candidate = f"{name}{season}这次预算{position + 12}元，请按时支付。"
                if index == 0 and position == 0:
                    candidate = SPECIAL_TEXT
                control = f"{name}{season}这里的解释清楚，我同意这段分析。"
                for pool, text in ((self.candidate_texts, candidate), (self.control_texts, control)):
                    pool.add(text)
                    row = self.raw_record(video, str(100000 + index * 100 + len(rows)), text)
                    rows.append(row)
                    self.original_by_text[text] = row
            for day in ("2023-12-31", "2026-01-02"):
                text = f"{name}这条在{day}发布，预算99元。"
                row = self.raw_record(video, str(100050 + index * 100 + len(rows)), text, day)
                rows.append(row)
                self.original_by_text[text] = row
                self.outside_texts.add(text)
            if index == 0:
                rows.append(dict(rows[0]))
                rows.append(self.raw_record(video, "1.23e+17", "这是非法评论标识的12元记录。"))
            path = self.source_root / f"video-{index}.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            raw = path.read_bytes()
            files.append({"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                          "git_blob_sha": hashlib.sha1(
                              b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest(),
                          "bytes": len(raw)})
        self.config = {
            "schema_version": 1, "source_id": "synthetic-fixture",
            "repo": "fixture/original-comments", "revision": REVISION,
            "capture_date": "2026-09-29",
            "license": {"status": "synthetic_test_fixture", "note": "Original test text only."},
            "files": files,
            "sampling": {"since": "2024-01-01", "until": "2026-01-01",
                         "seed": 20260929, "candidates": 14, "controls": 7},
        }
        self.config_path = self.root / "config.json"
        self.save_config()

    @staticmethod
    def raw_record(video, comment_id, text, day="2025-06-15"):
        return {"bv_id": video, "oid": "123456", "level": "1", "root_rpid": "0",
                "rpid": comment_id, "parent_rpid": "0", "uid": "123", "uname": "合成作者",
                "user_level": "1", "ctime": timestamp(day), "like": "0", "message": text}

    def save_config(self):
        self.config_path.write_text(json.dumps(self.config, ensure_ascii=False), encoding="utf-8")

    def rewrite_fixture_file(self, index, change):
        entry = self.config["files"][index]
        path = self.source_root / entry["path"]
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        change(rows)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        raw = path.read_bytes()
        entry.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                     git_blob_sha=hashlib.sha1(
                         b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest())
        self.save_config()

    def prepare(self, name="prepared"):
        output = self.root / name
        prepare_data(self.source_root, self.config_path, output)
        return output

    def reviewed(self, prepared):
        rows = read_jsonl(prepared / "review.jsonl")
        for row in rows:
            row["review"].update(status="approved", reviewer="fixture-reviewer",
                                 acceptable_outputs=[row["text"]], track="preservation",
                                 family="synthetic-source-preservation", context_required=False,
                                 notes="Synthetic fixture manually approved in this test.")
        return rows

    def export(self, prepared, rows, name="exported"):
        review_path = self.root / (name + ".review.jsonl")
        write_rows(review_path, rows)
        output = self.root / name
        export_review(prepared, review_path, output)
        return output, review_path

    def assert_export_rejected(self, prepared, rows, name="rejected-export"):
        review_path = self.root / (name + ".review.jsonl")
        write_rows(review_path, rows)
        output = self.root / name
        with self.assertRaises(ValueError):
            export_review(prepared, review_path, output)
        self.assertEqual(read_json(output / "run.json")["status"], "failed")
        self.assertFalse((output / "inputs.jsonl").exists())
        self.assertFalse((output / "cases.jsonl").exists())

    def test_prepare_preserves_sources_and_requires_review_before_inputs_exist(self):
        before = digest_tree(self.source_root)
        prepared = self.prepare()
        self.assertEqual(before, digest_tree(self.source_root))
        required = {"records.jsonl", "rejected.jsonl", "duplicates.jsonl", "groups.jsonl", "annotations.jsonl",
                    "review.jsonl", "review.md", "selection.json", "summary.json",
                    "manifest.json", "run.json"}
        self.assertTrue(required.issubset(path.name for path in prepared.iterdir()))
        self.assertEqual(read_json(prepared / "run.json")["status"], "complete")
        self.assertFalse((prepared / "inputs.jsonl").exists())
        self.assertFalse((prepared / "cases.jsonl").exists())
        records = read_jsonl(prepared / "records.jsonl")
        # Repeated identical IDs collapse, while their extra source receipt survives.
        # Out-of-window records remain available in the canonical capture.
        self.assertEqual(len(records), 84)
        special = [row for row in records if row["text"] == SPECIAL_TEXT]
        self.assertEqual(len(special), 1)
        self.assertEqual(len(special[0]["duplicate_sources"]), 1)
        self.assertEqual(len(read_jsonl(prepared / "rejected.jsonl")), 1)
        self.assertTrue(read_jsonl(prepared / "duplicates.jsonl"))
        for row in records:
            self.assertEqual(row["text"], self.original_by_text[row["text"]]["message"])
        for row in read_jsonl(prepared / "review.jsonl"):
            self.assertEqual(row["review"]["status"], "pending")
            self.assertEqual(row["review"]["acceptable_outputs"], [])
            self.assertEqual(row["review"]["reviewer"], "")
            self.assertIsNone(row["review"]["context_required"])
            self.assertNotIn("acceptable_outputs", {k: v for k, v in row.items() if k != "review"})

    def test_sampling_is_reproducible_date_filtered_and_balanced_by_video(self):
        first, second = self.prepare("first"), self.prepare("second")
        rows = read_jsonl(first / "review.jsonl")
        again = read_jsonl(second / "review.jsonl")
        self.assertEqual(rows, again)
        self.assertEqual(read_json(first / "selection.json"), read_json(second / "selection.json"))
        self.assertEqual(len(rows), 21)
        self.assertEqual(len({row["id"] for row in rows}), 21)
        self.assertFalse({row["text"] for row in rows} & self.outside_texts)
        chosen_candidates = [row for row in rows if row["text"] in self.candidate_texts]
        chosen_controls = [row for row in rows if row["text"] in self.control_texts]
        self.assertEqual(len(chosen_candidates), 14)
        self.assertEqual(len(chosen_controls), 7)
        for selection, quota in ((chosen_candidates, 2), (chosen_controls, 1)):
            counts = Counter(self.original_by_text[row["text"]]["bv_id"] for row in selection)
            self.assertEqual(len(counts), 7)
            self.assertEqual(set(counts.values()), {quota})
        self.assertTrue(all(not row["signals"] for row in chosen_controls))

    def test_sampling_includes_since_and_excludes_until_at_local_midnight(self):
        def set_boundary_dates(rows):
            rows[0]["ctime"] = timestamp("2024-01-01")
            rows[1]["ctime"] = timestamp("2026-01-01")

        self.rewrite_fixture_file(1, set_boundary_dates)
        self.config["sampling"].update(candidates=200, controls=200)
        self.save_config()
        prepared = self.prepare()
        selected = {row["text"] for row in read_jsonl(prepared / "review.jsonl")}
        self.assertIn("乙春这次预算12元，请按时支付。", selected)
        self.assertNotIn("乙春这里的解释清楚，我同意这段分析。", selected)
        # A pool shortage must not silently replace controls with candidate rows.
        summary = read_json(prepared / "summary.json")
        self.assertEqual(summary["selected_by_pool"], {"candidate": 35, "control": 34})

    def test_preparation_never_overwrites_an_existing_directory(self):
        prepared = self.prepare()
        before = digest_tree(prepared)
        with self.assertRaises(FileExistsError):
            prepare_data(self.source_root, self.config_path, prepared)
        self.assertEqual(before, digest_tree(prepared))

    def test_changed_source_file_fails_with_a_durable_run_record(self):
        source = self.source_root / self.config["files"][0]["path"]
        source.write_bytes(source.read_bytes() + b"\n")
        output = self.root / "bad-source"
        with self.assertRaises(ValueError):
            prepare_data(self.source_root, self.config_path, output)
        self.assertEqual(read_json(output / "run.json")["status"], "failed")
        self.assertFalse((output / "inputs.jsonl").exists())
        self.assertFalse((output / "review.jsonl").exists())

    def test_pending_review_cannot_export(self):
        prepared = self.prepare()
        self.assert_export_rejected(prepared, read_jsonl(prepared / "review.jsonl"))

    def test_approved_export_matches_existing_input_and_scoring_contracts(self):
        prepared = self.prepare()
        before = digest_tree(prepared)
        reviewed = self.reviewed(prepared)
        # Deliberately distinct reference text must remain offline rather than enter inputs.
        marker = "此句仅存在于离线人工参考。"
        reviewed[0]["review"].update(track="core_tn", family="reviewed-reference",
                                     acceptable_outputs=[marker])
        exported, review_path = self.export(prepared, reviewed)
        self.assertEqual(before, digest_tree(prepared))
        inputs = validate_inputs(exported / "inputs.jsonl")
        cases = read_jsonl(exported / "cases.jsonl")
        self.assertEqual(len(inputs), 21)
        self.assertEqual({row["id"] for row in inputs}, {row["id"] for row in cases})
        source_by_id = {row["id"]: row for row in reviewed}
        for row in inputs:
            self.assertEqual(set(row), {"id", "text"})
            self.assertEqual(row["text"], source_by_id[row["id"]]["text"])
        self.assertNotIn(marker, (exported / "inputs.jsonl").read_text(encoding="utf-8"))
        results = score(cases, [{"id": row["id"], "prediction": row["acceptable_outputs"][0],
                                "status": "ok"} for row in cases])
        self.assertEqual(results["overall"]["strict_correct"], 21)
        self.assertEqual(read_json(exported / "run.json")["status"], "complete")
        run = read_json(exported / "run.json")
        self.assertEqual(run["review_sha256"], hash_file(review_path))
        self.assertEqual(run["frozen_review_sha256"], hash_file(prepared / "review.jsonl"))

    def test_editing_annotations_preserves_the_frozen_review_and_exports(self):
        prepared = self.prepare()
        before = digest_tree(prepared)
        annotations = prepared / "annotations.jsonl"
        write_rows(annotations, self.reviewed(prepared))
        expected = dict(before)
        expected["annotations.jsonl"] = hash_file(annotations)
        output = self.root / "from-editable-annotations"
        export_review(prepared, annotations, output)
        self.assertEqual(expected, digest_tree(prepared))
        self.assertEqual(len(validate_inputs(output / "inputs.jsonl")), 21)
        self.assertEqual(read_json(output / "run.json")["review_sha256"], hash_file(annotations))

    def test_root_only_reply_context_is_disclosed_and_survives_provenance_export(self):
        def make_reply(rows):
            rows[1].update(level="2", root_rpid=rows[0]["rpid"], parent_rpid=rows[0]["rpid"])

        self.rewrite_fixture_file(1, make_reply)
        self.config["context"] = {
            "parent_semantics": "root_only",
            "note": "Synthetic collection saved the root comment, not the actual direct reply target.",
        }
        self.config["sampling"].update(candidates=100, controls=100)
        self.save_config()
        prepared = self.prepare()
        reply_text = "乙春这里的解释清楚，我同意这段分析。"
        review = next(row for row in read_jsonl(prepared / "review.jsonl")
                      if row["text"] == reply_text)
        self.assertEqual(review["context_kind"], "root_only")
        self.assertIn("direct_parent_not_preserved", review["quality_flags"])
        self.assertEqual([row["text"] for row in review["context"]],
                         ["乙春这次预算12元，请按时支付。"])
        markdown = (prepared / "review.md").read_text(encoding="utf-8")
        self.assertIn("真实直接回复对象未保留", markdown)
        exported, _ = self.export(prepared, self.reviewed(prepared))
        provenance = next(row for row in read_jsonl(exported / "provenance.jsonl")
                          if row["id"] == review["id"])
        self.assertEqual(provenance["context_kind"], "root_only")
        self.assertEqual(provenance["context"], review["context"])
        inference = next(row for row in validate_inputs(exported / "inputs.jsonl")
                         if row["id"] == review["id"])
        self.assertEqual(inference, {"id": review["id"], "text": reply_text})

    def test_unicode_line_and_paragraph_separators_survive_handwritten_review_and_legacy_reader(self):
        text = "这次预算12元\u2028这一句包含行分隔符\u2029这一句包含段落分隔符。"

        def put_separators_in_original(rows):
            rows[0]["message"] = text

        self.rewrite_fixture_file(1, put_separators_in_original)
        self.config["sampling"].update(candidates=100, controls=100)
        self.save_config()
        prepared = self.prepare()
        original = next(row for row in read_jsonl(prepared / "records.jsonl") if row["text"] == text)
        reviewed = self.reviewed(prepared)
        selected = next(row for row in reviewed if row["id"] == original["id"])
        self.assertEqual(selected["text"], text)
        annotations = prepared / "annotations.jsonl"
        # Simulate an editor writing literal Unicode rather than JSON \u escapes.
        write_rows(annotations, reviewed)
        handwritten = annotations.read_text(encoding="utf-8")
        self.assertIn("\u2028", handwritten)
        self.assertIn("\u2029", handwritten)
        self.assertEqual(read_jsonl(annotations), reviewed)
        output = self.root / "unicode-export"
        export_review(prepared, annotations, output)
        legacy_inputs = legacy_read_jsonl(output / "inputs.jsonl")
        legacy_cases = legacy_read_jsonl(output / "cases.jsonl")
        actual = next(row for row in legacy_inputs if row["id"] == original["id"])
        self.assertEqual(actual, {"id": original["id"], "text": text})
        self.assertEqual(legacy_inputs, validate_inputs(output / "inputs.jsonl"))
        references = next(row for row in legacy_cases if row["id"] == original["id"])
        self.assertEqual(references["acceptable_outputs"], [text])
        serialized = (output / "inputs.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("\u2028", serialized)
        self.assertNotIn("\u2029", serialized)
        self.assertEqual(score(legacy_cases, [
            {"id": row["id"], "prediction": row["text"], "status": "ok"}
            for row in legacy_inputs])["overall"]["strict_correct"], len(legacy_inputs))

    def test_excluded_and_uncertain_decisions_are_retained_and_not_scored(self):
        prepared = self.prepare()
        rows = self.reviewed(prepared)
        rows[0]["review"].update(status="excluded", acceptable_outputs=[], notes="Out of scope.")
        rows[1]["review"].update(status="uncertain", acceptable_outputs=[], notes="Needs more evidence.")
        exported, _ = self.export(prepared, rows)
        inputs = validate_inputs(exported / "inputs.jsonl")
        self.assertEqual(len(inputs), 19)
        self.assertNotIn(rows[0]["id"], {row["id"] for row in inputs})
        self.assertNotIn(rows[1]["id"], {row["id"] for row in inputs})
        retained = read_jsonl(exported / "review_decisions.jsonl")
        self.assertEqual({row["id"]: row["review"] for row in retained},
                         {row["id"]: row["review"] for row in rows})

    def test_invalid_approval_fields_cannot_export(self):
        prepared = self.prepare()
        mutations = [("reviewer", ""), ("acceptable_outputs", []),
                     ("acceptable_outputs", [""]), ("track", "invented"),
                     ("family", ""), ("context_required", True),
                     ("context_required", None), ("context_required", 0),
                     ("status", "not-a-status")]
        for index, (field, value) in enumerate(mutations):
            with self.subTest(field=field, value=value):
                rows = self.reviewed(prepared)
                rows[0]["review"][field] = value
                self.assert_export_rejected(prepared, rows, f"invalid-{index}")

    def test_exclusion_and_uncertainty_require_reviewer_and_reason(self):
        prepared = self.prepare()
        for status in ("excluded", "uncertain"):
            for field in ("reviewer", "notes"):
                with self.subTest(status=status, field=field):
                    rows = self.reviewed(prepared)
                    rows[0]["review"].update(status=status, notes="Needs review evidence.")
                    rows[0]["review"][field] = ""
                    self.assert_export_rejected(prepared, rows, status + "-no-" + field)

    def test_preservation_reference_must_retain_exact_text(self):
        prepared = self.prepare()
        rows = self.reviewed(prepared)
        rows[0]["review"]["acceptable_outputs"] = [rows[0]["text"] + "已经改写。"]
        self.assert_export_rejected(prepared, rows)

    def test_review_cannot_change_text_id_source_or_add_gold_outside_review(self):
        prepared = self.prepare()
        for field in ("text", "id", "source", "context", "signals", "quality_flags",
                      "selection_pool", "acceptable_outputs"):
            with self.subTest(field=field):
                rows = self.reviewed(prepared)
                if field == "text":
                    rows[0][field] += " 标准答案是十二元。"
                elif field == "id":
                    rows[0][field] = "changed-source-id"
                elif field == "acceptable_outputs":
                    rows[0][field] = ["伪装在审核对象之外的参考。"]
                else:
                    rows[0][field] = {"tampered": True}
                self.assert_export_rejected(prepared, rows, "tampered-" + field)

    def test_missing_duplicate_or_added_review_items_cannot_export(self):
        prepared = self.prepare()
        rows = self.reviewed(prepared)
        modified = copy.deepcopy(rows[0])
        modified["id"] = "unregistered-id"
        for index, altered in enumerate((rows[:-1], rows + [rows[0]], rows + [modified])):
            with self.subTest(kind=index):
                self.assert_export_rejected(prepared, altered, f"bad-roster-{index}")

    def test_corrupted_prepared_artifacts_or_manifest_cannot_export(self):
        for name in ("records.jsonl", "review.jsonl", "manifest.json"):
            with self.subTest(file=name):
                prepared = self.prepare("prepared-" + name)
                rows = self.reviewed(prepared)
                path = prepared / name
                if name == "manifest.json":
                    path.write_text("{}\n", encoding="utf-8")
                else:
                    path.write_bytes(path.read_bytes() + b"\n")
                self.assert_export_rejected(prepared, rows, "corrupt-" + name)

    def test_export_never_overwrites_prepared_or_an_existing_export(self):
        prepared = self.prepare()
        rows = self.reviewed(prepared)
        exported, review_path = self.export(prepared, rows)
        for destination in (prepared, exported):
            with self.subTest(destination=destination.name):
                before = digest_tree(destination)
                with self.assertRaises((FileExistsError, ValueError)):
                    export_review(prepared, review_path, destination)
                self.assertEqual(before, digest_tree(destination))


if __name__ == "__main__":
    unittest.main()
