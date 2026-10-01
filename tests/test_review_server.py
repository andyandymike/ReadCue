"""Real loopback HTTP checks for the reviewer, using only synthetic comments."""
import copy
from concurrent.futures import ThreadPoolExecutor
import http.client
import hashlib
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from readcue import review_server
from readcue.artifacts import hash_file, read_json, read_jsonl
from readcue.review_server import ReviewError, create_server
import test_data_pipeline as pipeline_fixture


UNICODE_TEXT = "  预算１２元，型号 A7。\n换行保持\u2028行分隔符\u2029段落分隔符也保持。  "
READING_TEXT = "🤦‍♂️AI 与 AI 用 tomcat9，c++。"
WHOLE_READING_TEXT = "项目用jdk8tomcat9。"


class ReviewServerTests(unittest.TestCase):
    def setUp(self):
        # Reuse fixture construction without inheriting/rerunning pipeline tests.
        self.fixture = pipeline_fixture.DataPipelineTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

        def add_unicode(rows):
            rows[0]["message"] = UNICODE_TEXT
            rows[2]["message"] = READING_TEXT
            rows[4]["message"] = WHOLE_READING_TEXT

        self.fixture.rewrite_fixture_file(1, add_unicode)
        self.fixture.config["sampling"].update(candidates=100, controls=100)
        self.fixture.save_config()
        self.prepared = self.fixture.prepare()
        self.annotations = self.prepared / "annotations.jsonl"
        self.fixed_hashes = {name: item["sha256"] for name, item in
                            read_json(self.prepared / "manifest.json")["files"].items()}
        self.fixed_hashes.update({name: hash_file(self.prepared / name)
                                 for name in ("manifest.json", "run.json")})
        self.server = None
        self.server_thread = None
        self.addCleanup(self.stop_server)
        self.start_server()

    def start_server(self, annotations=None):
        self.server = create_server(self.prepared, annotations=annotations, port=0)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.server_thread.start()
        parsed = urlsplit(self.server.base_url)
        self.origin = f"{parsed.scheme}://{parsed.netloc}"
        self.prefix = parsed.path

    def stop_server(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server_thread.join(timeout=3)
            if self.server_thread.is_alive():
                raise AssertionError("Review HTTP server did not stop")
            self.server = None

    def request(self, method, suffix="api/state", payload=None, *, headers=None, path=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers = {"Origin": self.origin}
        if data is not None:
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path or self.prefix + suffix, body=data, headers=request_headers)
            response = connection.getresponse()
            raw = response.read()
            content_type = response.getheader("Content-Type", "")
            decoded = json.loads(raw) if "application/json" in content_type else raw.decode("utf-8")
            return response.status, decoded, dict(response.getheaders())
        finally:
            connection.close()

    def state(self):
        status, body, _ = self.request("GET")
        self.assertEqual(status, 200, body)
        return body

    @staticmethod
    def approved_review(item):
        review = copy.deepcopy(item["review"])
        review.update(status="approved", reviewer="synthetic-reviewer",
                      acceptable_outputs=[item["text"]], track="preservation",
                      family="synthetic-preservation", context_required=False,
                      notes="Synthetic fixture decision.")
        return review

    def payload(self, state, item=None, review=None):
        item = item or state["items"][0]
        return {"id": item["id"], "review": review or self.approved_review(item),
                "version": state["version"], "token": self.server.token}

    def assert_fixed_capture_unchanged(self):
        self.assertEqual({name: hash_file(self.prepared / name) for name in self.fixed_hashes},
                         self.fixed_hashes)

    def assert_rejected_without_writing(self, payload, expected=400, **kwargs):
        before = self.annotations.read_bytes()
        status, body, _ = self.request("POST", "api/save", payload, **kwargs)
        self.assertEqual(status, expected, body)
        self.assertIsInstance(body.get("error"), str)
        self.assertEqual(before, self.annotations.read_bytes())
        self.assert_fixed_capture_unchanged()

    def suggestion_envelope(self):
        item = read_jsonl(self.prepared / "review.jsonl")[0]
        return {"schema_version": 1, "prepared_manifest_sha256": hash_file(self.prepared / "manifest.json"),
                "generator": "Codex assistant", "human_reviewed": False,
                "policy_version": "bilibili-ai-review-v1", "created_at_utc": "2026-09-29T09:00:00+00:00",
                "items": [{"id": item["id"], "text_sha256": hashlib.sha256(item["text"].encode("utf-8")).hexdigest(),
                           "decision": "keep", "acceptable_outputs": [item["text"]], "track": "preservation",
                           "family": "AI-only-family", "context_required": False,
                           "reason": "Synthetic machine suggestion, not a human decision.", "flags": ["fixture"]}]}

    def write_suggestions(self, envelope):
        path = self.prepared / "ai-suggestions.json"
        path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        return path

    def reading_suggestion_envelope(self):
        envelope = self.suggestion_envelope()
        envelope.update(schema_version=2, policy_version="bilibili-ai-review-v3")
        item = next(row for row in read_jsonl(self.prepared / "review.jsonl")
                    if row["text"] == READING_TEXT)
        suggestion = envelope["items"][0]
        suggestion.update(id=item["id"],
                          text_sha256=hashlib.sha256(READING_TEXT.encode("utf-8")).hexdigest(),
                          acceptable_outputs=[READING_TEXT])
        # Literal codepoint offsets make the astral emoji and repeated AI binding explicit.
        spans = [(0, 4, "🤦‍♂️", 1, "omit", "不读"),
                 (4, 6, "AI", 1, "letters_en", "A、I（英文字母名）"),
                 (9, 11, "AI", 2, "letters_en", "A、I（英文字母名）"),
                 (14, 20, "tomcat", 1, "word_en", "tomcat（英文词）"),
                 (20, 21, "9", 1, "number_zh", "九"),
                 (22, 25, "c++", 1, "mixed", "C plus plus")]
        suggestion["reading_audit"] = {
            "status": "needs_reading_annotation", "summary": "Synthetic reading proposals only.",
            "issues": [], "spans": [
                {"start": start, "end": end, "surface": surface, "occurrence": occurrence,
                 "mode": mode, "reading": reading, "reason": "Synthetic span proposal."}
                for start, end, surface, occurrence, mode, reading in spans]}
        return envelope

    def start_v5_round(self, *, previously_approved=False):
        self.stop_server()
        envelope = self.reading_suggestion_envelope()
        envelope["policy_version"] = "bilibili-ai-review-v5"
        suggestion = envelope["items"][0]
        suggestion.update(decision="edit", acceptable_outputs=["A I 与 A I 用 tomcat 九，C plus plus。"],
                          track="context_reading", family="英文与缩写")
        rows = read_jsonl(self.annotations)
        target = next(row for row in rows if row["id"] == suggestion["id"])
        if previously_approved:
            target["review"] = self.approved_review(target)
            pipeline_fixture.write_rows(self.annotations, rows)
        self.write_suggestions(envelope)
        self.start_server()
        return envelope, target

    def inline_review(self, item, suggestion):
        review = self.approved_review(item)
        review.update(acceptable_outputs=suggestion["acceptable_outputs"], track=suggestion["track"],
                      family=suggestion["family"], notes="Human checked the complete candidate.")
        return review

    def start_two_item_v5_round(self):
        envelope, target = self.start_v5_round()
        self.stop_server()
        unrelated = self.suggestion_envelope()["items"][0]
        unrelated["reading_audit"] = {"status": "text_only_ok", "summary": "Synthetic unchanged text.",
                                      "spans": [], "issues": []}
        self.assertNotEqual(unrelated["id"], target["id"])
        envelope["items"].append(unrelated)
        self.write_suggestions(envelope)
        self.start_server()
        return envelope, target

    def archive_suggestions(self):
        body = (self.prepared / "ai-suggestions.json").read_bytes()
        directory = self.prepared / "ai-suggestions.history"
        directory.mkdir(exist_ok=True)
        path = directory / (review_server._sha(body) + ".json")
        path.write_bytes(body)
        return path

    def test_loopback_entry_serves_html_and_real_pending_batch(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        status, html, headers = self.request("GET", "")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("<html", html.lower())
        state = self.state()
        self.assertIs(state["preview"], False)
        self.assertEqual(state["token"], self.server.token)
        self.assertEqual(Path(state["annotation_path"]).resolve(), self.annotations.resolve())
        self.assertIn("batch", state)
        self.assertEqual(len(state["items"]), 70)
        self.assertTrue(all(item["review"]["status"] == "pending" for item in state["items"]))
        self.assertIsNone(state["suggestions"])
        self.assertIsNone(state["suggestions_sha256"])
        self.assertEqual(state["suggestion_review_state"], {})
        self.assertEqual(state["suggestion_review_choices"], {})
        self.assert_fixed_capture_unchanged()

    def test_v5_explicit_confirmation_and_amendment_survive_reload_and_restart(self):
        envelope, target = self.start_v5_round(previously_approved=True)
        ident = target["id"]
        before = self.annotations.read_bytes()
        state = self.state()
        self.assertEqual(state["suggestion_review_state"], {})
        self.assertEqual(next(row for row in state["items"] if row["id"] == ident), target)
        self.assertEqual(self.annotations.read_bytes(), before)
        review = self.inline_review(target, envelope["items"][0])
        status, result, _ = self.request("POST", "api/save", self.payload(state, target, review))
        self.assertEqual(status, 200, result)
        self.assertEqual(result["suggestion_review_state"], {ident: "reviewed"})
        self.assertEqual(result["suggestion_review_choices"], {})
        self.assertEqual(self.state()["suggestion_review_state"], {ident: "reviewed"})

        amended = copy.deepcopy(review)
        amended.update(acceptable_outputs=["A I 与 A I 用 tomcat 九，C 加加。"], notes="My amended candidate.")
        status, result, _ = self.request("POST", "api/save", self.payload(self.state(), target, amended))
        self.assertEqual(status, 200, result)
        self.assertEqual(result["suggestion_review_state"], {ident: "reviewed"})
        self.stop_server()
        self.start_server()
        resumed = self.state()
        self.assertEqual(resumed["suggestion_review_state"], {ident: "reviewed"})
        saved = next(row for row in resumed["items"] if row["id"] == ident)
        self.assertEqual(saved["review"], amended)
        self.assertEqual(set(saved["review"]), review_server.REVIEW_KEYS)
        latest = sorted(Path(self.server.history_path).glob("*/change.json"))[-1]
        receipt = read_json(latest)
        self.assertEqual(receipt["saved_review_sha256"], review_server._sha(review_server._json_bytes(amended)))
        self.assertEqual(receipt["available_ai_suggestion"]["sha256"], resumed["suggestions_sha256"])
        self.assertEqual(receipt["available_ai_suggestion"]["policy_version"], "bilibili-ai-review-v5")
        self.assertEqual(receipt["available_ai_suggestion"]["item_sha256"],
                         review_server._sha(review_server._json_bytes(envelope["items"][0])))
        self.assert_fixed_capture_unchanged()

    def test_v6_item_hash_keeps_unrelated_confirmations_but_changed_item_needs_review(self):
        envelope, target = self.start_two_item_v5_round()
        ident = target["id"]
        review = self.inline_review(target, envelope["items"][0])
        payload = self.payload(self.state(), target, review)
        payload["review_choice"] = "edit"
        status, result, _ = self.request("POST", "api/save", payload)
        self.assertEqual(status, 200, result)
        before = self.annotations.read_bytes()
        self.stop_server()
        envelope["policy_version"] = "bilibili-ai-review-v6"
        envelope["items"][1]["reason"] = "Only this unrelated suggestion changed."
        self.write_suggestions(envelope)
        self.start_server()
        state = self.state()
        self.assertEqual(state["suggestion_review_state"], {ident: "reviewed"})
        self.assertEqual(state["suggestion_review_choices"], {ident: "edit"})
        self.assertFalse((self.prepared / "ai-suggestions.history").exists())
        self.assertEqual(self.annotations.read_bytes(), before)
        self.stop_server()
        envelope["items"][0]["acceptable_outputs"] = ["A I 与 A I 用 tomcat 九，C 加加。"]
        self.write_suggestions(envelope)
        self.start_server()
        self.assertEqual(self.state()["suggestion_review_state"], {})
        self.assertEqual(self.state()["suggestion_review_choices"], {})
        self.assertEqual(self.annotations.read_bytes(), before)
        updated = self.inline_review(target, envelope["items"][0])
        status, result, _ = self.request("POST", "api/save", self.payload(self.state(), target, updated))
        self.assertEqual(status, 200, result)
        self.assertEqual(result["suggestion_review_state"], {ident: "reviewed"})
        latest = sorted(Path(self.server.history_path).glob("*/change.json"))[-1]
        self.assertEqual(read_json(latest)["available_ai_suggestion"]["policy_version"], "bilibili-ai-review-v6")
        self.assert_fixed_capture_unchanged()

    def test_v6_verified_archive_carries_legacy_confirmation_and_draft_choice(self):
        for saved_status, choice, expected in (("approved", "edit", "reviewed"),
                                               ("pending", "uncertain", "draft")):
            with self.subTest(saved_status=saved_status):
                envelope, target = self.start_two_item_v5_round()
                ident = target["id"]
                review = self.inline_review(target, envelope["items"][0])
                review.update(status=saved_status, acceptable_outputs=["My chosen complete reading."],
                              notes="Preserve the user's amendment across unrelated changes.")
                payload = self.payload(self.state(), target, review)
                payload["review_choice"] = choice
                status, result, _ = self.request("POST", "api/save", payload)
                self.assertEqual(status, 200, result)
                receipt_path = sorted(Path(self.server.history_path).glob("*/change.json"))[-1]
                receipt = read_json(receipt_path)
                receipt["available_ai_suggestion"].pop("item_sha256")
                receipt_path.write_bytes(review_server._json_bytes(receipt))
                archived = self.archive_suggestions()
                protected = {path: path.read_bytes() for path in (self.annotations, receipt_path, archived)}
                self.stop_server()
                envelope["policy_version"] = "bilibili-ai-review-v6"
                envelope["items"][1]["reason"] = "New prefix rule applies only to another record."
                self.write_suggestions(envelope)
                self.start_server()
                state = self.state()
                self.assertEqual(state["suggestion_review_state"], {ident: expected})
                self.assertEqual(state["suggestion_review_choices"], {ident: choice})
                self.assertEqual(next(row for row in state["items"] if row["id"] == ident)["review"], review)
                self.stop_server()
                self.start_server()
                self.assertEqual(self.state()["suggestion_review_choices"], {ident: choice})
                self.assertEqual({path: path.read_bytes() for path in protected}, protected)
                # Even a valid legacy archive cannot acknowledge a changed target.
                self.stop_server()
                envelope["items"][0]["reason"] = "This record now needs another confirmation."
                self.write_suggestions(envelope)
                self.start_server()
                self.assertEqual(self.state()["suggestion_review_state"], {})
                self.assertEqual(self.state()["suggestion_review_choices"], {})
        self.assert_fixed_capture_unchanged()

    def test_v6_missing_tampered_or_invalid_archives_never_acknowledge_legacy_save(self):
        envelope, target = self.start_two_item_v5_round()
        review = self.inline_review(target, envelope["items"][0])
        status, result, _ = self.request("POST", "api/save", self.payload(self.state(), target, review))
        self.assertEqual(status, 200, result)
        receipt_path = sorted(Path(self.server.history_path).glob("*/change.json"))[-1]
        receipt = read_json(receipt_path)
        receipt["available_ai_suggestion"].pop("item_sha256")
        receipt_path.write_bytes(review_server._json_bytes(receipt))
        archived = self.archive_suggestions()
        archive_body = archived.read_bytes()
        before = self.annotations.read_bytes()
        self.stop_server()
        previous = copy.deepcopy(envelope)
        envelope["policy_version"] = "bilibili-ai-review-v6"
        envelope["items"][1]["reason"] = "Unrelated new suggestion."
        self.write_suggestions(envelope)
        self.start_server()
        self.assertEqual(self.state()["suggestion_review_state"], {target["id"]: "reviewed"})
        for mutation in ("missing", "tampered", "oversized", "manifest", "text", "v4", "path", "item_hash"):
            with self.subTest(mutation=mutation):
                archived.write_bytes(archive_body)
                current_receipt = copy.deepcopy(receipt)
                if mutation == "missing":
                    archived.unlink()
                elif mutation == "tampered":
                    archived.write_bytes(archive_body + b" ")
                elif mutation == "oversized":
                    archived.write_bytes(b"x" * (review_server.MAX_SUGGESTION_BYTES + 1))
                elif mutation in {"manifest", "text", "v4"}:
                    invalid = copy.deepcopy(previous)
                    if mutation == "manifest":
                        invalid["prepared_manifest_sha256"] = "0" * 64
                    elif mutation == "text":
                        invalid["items"][0]["text_sha256"] = "0" * 64
                    else:
                        invalid["policy_version"] = "bilibili-ai-review-v4"
                        current_receipt["available_ai_suggestion"]["policy_version"] = invalid["policy_version"]
                    invalid_body = review_server._json_bytes(invalid)
                    digest = review_server._sha(invalid_body)
                    (archived.parent / (digest + ".json")).write_bytes(invalid_body)
                    current_receipt["available_ai_suggestion"]["sha256"] = digest
                elif mutation == "path":
                    current_receipt["available_ai_suggestion"]["sha256"] = "../outside"
                else:
                    # A present but invalid item hash must not fall back to the archive.
                    current_receipt["available_ai_suggestion"]["item_sha256"] = "0" * 64
                receipt_path.write_bytes(review_server._json_bytes(current_receipt))
                self.assertEqual(self.state()["suggestion_review_state"], {})
                self.assertEqual(self.state()["suggestion_review_choices"], {})
                self.assertEqual(self.annotations.read_bytes(), before)
        self.assert_fixed_capture_unchanged()

    def test_v5_drafts_persist_and_uncertain_or_excluded_saves_count_as_reviewed(self):
        envelope, target = self.start_v5_round()
        ident = target["id"]
        review = self.inline_review(target, envelope["items"][0])
        review.update(status="pending", notes="Draft with my own reading.", acceptable_outputs=["My partial draft."])
        status, result, _ = self.request("POST", "api/save", self.payload(self.state(), target, review))
        self.assertEqual(status, 200, result)
        self.assertEqual(result["suggestion_review_state"], {ident: "draft"})
        self.stop_server()
        self.start_server()
        state = self.state()
        self.assertEqual(state["suggestion_review_state"], {ident: "draft"})
        self.assertEqual(next(row for row in state["items"] if row["id"] == ident)["review"], review)
        for decision in ("uncertain", "excluded"):
            with self.subTest(decision=decision):
                review.update(status=decision, notes="Human chose this routing.")
                status, result, _ = self.request("POST", "api/save", self.payload(self.state(), target, review))
                self.assertEqual(status, 200, result)
                self.assertEqual(result["suggestion_review_state"], {ident: "reviewed"})
                self.assertEqual(self.state()["suggestion_review_state"], {ident: "reviewed"})
        self.assert_fixed_capture_unchanged()

    def test_new_suggestion_hash_starts_a_new_unreviewed_round(self):
        envelope, target = self.start_v5_round(previously_approved=True)
        review = self.inline_review(target, envelope["items"][0])
        payload = self.payload(self.state(), target, review)
        payload["review_choice"] = "edit"
        status, result, _ = self.request("POST", "api/save", payload)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["suggestion_review_state"], {target["id"]: "reviewed"})
        self.assertEqual(result["suggestion_review_choices"], {target["id"]: "edit"})
        before = self.annotations.read_bytes()
        self.stop_server()
        envelope["items"][0]["reason"] = "A new review round with a different candidate explanation."
        self.write_suggestions(envelope)
        self.start_server()
        self.assertEqual(self.state()["suggestion_review_state"], {})
        self.assertEqual(self.state()["suggestion_review_choices"], {})
        self.assertEqual(self.annotations.read_bytes(), before)

    def test_v5_pending_form_choice_is_separate_from_review_and_survives_restart(self):
        envelope, target = self.start_v5_round()
        ident = target["id"]
        review = self.inline_review(target, envelope["items"][0])
        review.update(status="pending", notes="My saved reading draft.")
        for choice in ("keep", "uncertain", "excluded", "", "edit"):
            with self.subTest(choice=choice):
                payload = self.payload(self.state(), target, review)
                payload["review_choice"] = choice
                status, result, _ = self.request("POST", "api/save", payload)
                self.assertEqual(status, 200, result)
                self.assertEqual(result["suggestion_review_state"], {ident: "draft"})
                self.assertEqual(result["suggestion_review_choices"], {ident: choice})
                self.assertEqual(set(result["item"]["review"]), review_server.REVIEW_KEYS)
                self.assertEqual(result["item"]["review"], review)
                self.assertEqual(self.state()["suggestion_review_choices"], {ident: choice})
        self.stop_server()
        self.start_server()
        state = self.state()
        self.assertEqual(state["suggestion_review_state"], {ident: "draft"})
        self.assertEqual(state["suggestion_review_choices"], {ident: "edit"})
        latest = sorted(Path(self.server.history_path).glob("*/change.json"))[-1]
        self.assertEqual(read_json(latest)["review_choice"], "edit")
        saved = next(row for row in state["items"] if row["id"] == ident)
        self.assertEqual(saved["review"], review)
        self.assertNotIn("review_choice", saved)
        self.assert_fixed_capture_unchanged()

    def test_invalid_form_choice_types_and_nonpending_mismatches_are_rejected(self):
        envelope, target = self.start_v5_round()
        review = self.inline_review(target, envelope["items"][0])
        for choice in (None, True, 1, [], {}, "unknown", "", "keep", "uncertain", "excluded"):
            with self.subTest(choice=choice):
                payload = self.payload(self.state(), target, review)
                payload["review_choice"] = choice
                self.assert_rejected_without_writing(payload)
        # Each valid final status has one consistent choice. Legacy callers may omit it.
        for choice in ("keep", "edit", "uncertain", "excluded"):
            with self.subTest(valid_choice=choice):
                valid = self.approved_review(target) if choice == "keep" else copy.deepcopy(review)
                valid["status"] = "approved" if choice in {"keep", "edit"} else choice
                payload = self.payload(self.state(), target, valid)
                payload["review_choice"] = choice
                status, result, _ = self.request("POST", "api/save", payload)
                self.assertEqual(status, 200, result)
                self.assertEqual(result["suggestion_review_choices"], {target["id"]: choice})
                payload = self.payload(self.state(), target, valid)
                payload["review_choice"] = "edit" if choice != "edit" else "keep"
                self.assert_rejected_without_writing(payload)
        self.assert_fixed_capture_unchanged()

    def test_latest_committed_receipt_prevents_older_match_after_a_row_is_restored(self):
        envelope, target = self.start_v5_round(previously_approved=True)
        ident = target["id"]
        first_review = self.inline_review(target, envelope["items"][0])
        status, result, _ = self.request("POST", "api/save", self.payload(self.state(), target, first_review))
        self.assertEqual(status, 200, result)
        first_rows = read_jsonl(self.annotations)
        next_review = copy.deepcopy(first_review)
        next_review["notes"] = "A later amendment must invalidate an older matching receipt."
        payload = self.payload(self.state(), target, next_review)
        payload["review_choice"] = "edit"
        status, result, _ = self.request("POST", "api/save", payload)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["suggestion_review_state"], {ident: "reviewed"})
        pipeline_fixture.write_rows(self.annotations, first_rows)
        self.assertEqual(self.state()["suggestion_review_state"], {})
        self.assertEqual(self.state()["suggestion_review_choices"], {})
        # Even an older matching receipt cannot supersede a latest legacy receipt.
        latest = sorted(Path(self.server.history_path).glob("*/change.json"))[-1]
        receipt = read_json(latest)
        receipt.pop("saved_review_sha256")
        latest.write_bytes(review_server._json_bytes(receipt))
        self.assertEqual(self.state()["suggestion_review_state"], {})
        self.assert_fixed_capture_unchanged()

    def test_failed_and_prepared_receipts_never_acknowledge_a_suggestion(self):
        envelope, target = self.start_v5_round(previously_approved=True)
        review = self.inline_review(target, envelope["items"][0])
        actual_replace = review_server.os.replace

        def fail_annotations(source, destination):
            if Path(destination).resolve() == self.annotations.resolve():
                raise OSError("Synthetic annotation replacement failure")
            return actual_replace(source, destination)

        with patch.object(review_server.os, "replace", side_effect=fail_annotations):
            self.assert_rejected_without_writing(self.payload(self.state(), target, review), expected=500)
        self.assertEqual(self.state()["suggestion_review_state"], {})
        failed = sorted(Path(self.server.history_path).glob("*/change.json"))[-1]
        self.assertEqual(read_json(failed)["status"], "failed")

        def fail_commit_receipt(source, destination):
            if Path(destination).name == "change.json":
                raise OSError("Synthetic receipt completion failure")
            return actual_replace(source, destination)

        with patch.object(review_server.os, "replace", side_effect=fail_commit_receipt):
            status, result, _ = self.request("POST", "api/save", self.payload(self.state(), target, review))
        self.assertEqual(status, 500, result)
        prepared = sorted(Path(self.server.history_path).glob("*/change.json"))[-1]
        self.assertEqual(read_json(prepared)["status"], "prepared")
        # The actual row was saved, but an uncommitted receipt cannot acknowledge it.
        saved = next(row for row in self.state()["items"] if row["id"] == target["id"])
        self.assertEqual(saved["review"], review)
        self.assertEqual(self.state()["suggestion_review_state"], {})
        self.assert_fixed_capture_unchanged()

    def test_legacy_receipts_and_invalid_receipt_bindings_do_not_count_as_reviewed(self):
        envelope, target = self.start_v5_round()
        review = self.inline_review(target, envelope["items"][0])
        status, result, _ = self.request("POST", "api/save", self.payload(self.state(), target, review))
        self.assertEqual(status, 200, result)
        receipt_path = next(Path(self.server.history_path).glob("*/change.json"))
        receipt = read_json(receipt_path)
        receipt.pop("saved_review_sha256")
        receipt_path.write_bytes(review_server._json_bytes(receipt))
        self.assertEqual(self.state()["suggestion_review_state"], {})
        receipt["annotation_path"] = str(self.prepared / "different-annotations.jsonl")
        receipt_path.write_bytes(review_server._json_bytes(receipt))
        status, error, _ = self.request("GET")
        self.assertEqual(status, 500, error)
        self.assertIn("历史", error["error"])
        self.assert_fixed_capture_unchanged()

    def test_ai_suggestions_remain_separate_until_an_explicit_human_save(self):
        self.stop_server()
        before = self.annotations.read_bytes()
        envelope = self.suggestion_envelope()
        sidecar = self.write_suggestions(envelope)
        self.start_server()
        state = self.state()
        self.assertEqual(state["suggestions"], envelope)
        self.assertEqual(state["suggestions_sha256"], hash_file(sidecar))
        self.assertEqual(state["items"], read_jsonl(self.annotations))
        self.assertEqual(before, self.annotations.read_bytes())
        self.assertTrue(all(item["review"]["status"] == "pending" for item in state["items"]))
        self.assertTrue(all(item["review"]["family"] == "" for item in state["items"]))
        self.assertFalse((self.prepared / "cases.jsonl").exists())
        status, result, _ = self.request("POST", "api/save", self.payload(state))
        self.assertEqual(status, 200, result)
        self.assertEqual(result["item"]["review"]["reviewer"], "synthetic-reviewer")
        self.assertEqual(set(result["item"]["review"]), review_server.REVIEW_KEYS)
        self.assertEqual(result["item"]["review"]["family"], "synthetic-preservation")
        change = next(Path(self.server.history_path).glob("*/change.json"))
        audit = read_json(change)
        self.assertEqual(audit["available_ai_suggestion"]["sha256"], hash_file(sidecar))
        self.assertIs(audit["available_ai_suggestion"]["human_reviewed"], False)
        self.assertEqual(change.with_name("before.jsonl").read_bytes(), before)
        self.assertEqual(read_json(sidecar), envelope)
        self.assertEqual(sum(item["review"]["status"] == "approved" for item in self.state()["items"]), 1)
        self.assert_fixed_capture_unchanged()

    def test_suggestion_loading_preserves_saved_decisions_and_nonempty_drafts(self):
        self.stop_server()
        rows = read_jsonl(self.annotations)
        rows[0]["review"] = self.approved_review(rows[0])
        rows[1]["review"].update(reviewer="draft-author", notes="Do not replace this human draft.", family="human-family")
        pipeline_fixture.write_rows(self.annotations, rows)
        before = self.annotations.read_bytes()
        envelope = self.suggestion_envelope()
        second = copy.deepcopy(envelope["items"][0])
        second.update(id=rows[1]["id"], text_sha256=hashlib.sha256(rows[1]["text"].encode("utf-8")).hexdigest(),
                      acceptable_outputs=[rows[1]["text"]])
        envelope["items"].append(second)
        self.write_suggestions(envelope)
        self.start_server()
        self.assertEqual(self.state()["items"], rows)
        self.assertEqual(self.annotations.read_bytes(), before)
        self.assertFalse(Path(self.server.history_path).exists())

    def test_v2_suggestions_load_and_preserve_policy_version_in_save_history(self):
        self.stop_server()
        before = self.annotations.read_bytes()
        envelope = self.suggestion_envelope()
        envelope["policy_version"] = "bilibili-ai-review-v2"
        sidecar = self.write_suggestions(envelope)
        self.start_server()
        state = self.state()
        self.assertEqual(state["suggestions"]["schema_version"], 1)
        self.assertEqual(state["suggestions"]["policy_version"], "bilibili-ai-review-v2")
        self.assertEqual(state["suggestions_sha256"], hash_file(sidecar))
        self.assertEqual(before, self.annotations.read_bytes())
        status, result, _ = self.request("POST", "api/save", self.payload(state))
        self.assertEqual(status, 200, result)
        self.assertEqual(result["item"]["review"]["reviewer"], "synthetic-reviewer")
        change = next(Path(self.server.history_path).glob("*/change.json"))
        audit = read_json(change)
        self.assertEqual(audit["status"], "committed")
        self.assertEqual(audit["available_ai_suggestion"]["policy_version"], "bilibili-ai-review-v2")
        self.assertEqual(audit["available_ai_suggestion"]["sha256"], hash_file(sidecar))
        self.assertEqual(change.with_name("before.jsonl").read_bytes(), before)
        self.assertEqual(read_json(sidecar), envelope)
        self.assert_fixed_capture_unchanged()

    def test_v3_reading_audit_stays_separate_from_human_reviews_and_save_history(self):
        self.stop_server()
        envelope = self.reading_suggestion_envelope()
        ident = envelope["items"][0]["id"]
        rows = read_jsonl(self.annotations)
        target = next(row for row in rows if row["id"] == ident)
        target["review"].update(reviewer="draft-author", family="human-draft",
                                notes="My draft must not become an AI decision.")
        approved = next(row for row in rows if row["id"] != ident)
        approved["review"] = self.approved_review(approved)
        pipeline_fixture.write_rows(self.annotations, rows)
        before = self.annotations.read_bytes()
        sidecar = self.write_suggestions(envelope)
        sidecar_bytes = sidecar.read_bytes()
        self.start_server()

        state = self.state()
        self.assertEqual(state["suggestions"], envelope)
        self.assertEqual(state["suggestions_sha256"], hash_file(sidecar))
        self.assertEqual(state["items"], rows)
        self.assertEqual(self.annotations.read_bytes(), before)
        self.assertFalse(Path(self.server.history_path).exists())
        self.assertFalse((self.prepared / "cases.jsonl").exists())
        self.assertFalse((self.prepared / "inputs.jsonl").exists())

        target = next(row for row in state["items"] if row["id"] == ident)
        payload = self.payload(state, target)
        injected = copy.deepcopy(payload)
        injected["review"]["reading_audit"] = envelope["items"][0]["reading_audit"]
        self.assert_rejected_without_writing(injected)
        status, result, _ = self.request("POST", "api/save", payload)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["item"]["review"], payload["review"])
        self.assertEqual(len(result["item"]["review"]), 7)
        self.assertEqual(set(result["item"]["review"]), review_server.REVIEW_KEYS)
        self.assertNotIn("reading_audit", result["item"])
        for row in read_jsonl(self.annotations):
            if row["id"] == ident:
                self.assertEqual(row["review"], payload["review"])
            else:
                self.assertEqual(row, next(old for old in rows if old["id"] == row["id"]))
        change = next(Path(self.server.history_path).glob("*/change.json"))
        audit = read_json(change)
        self.assertEqual(audit["status"], "committed")
        self.assertEqual(audit["available_ai_suggestion"]["policy_version"], "bilibili-ai-review-v3")
        self.assertEqual(audit["available_ai_suggestion"]["sha256"], hash_file(sidecar))
        self.assertIs(audit["available_ai_suggestion"]["human_reviewed"], False)
        self.assertEqual(change.with_name("before.jsonl").read_bytes(), before)
        self.assertEqual(sidecar.read_bytes(), sidecar_bytes)
        self.assert_fixed_capture_unchanged()

    def test_v4_whole_reading_candidates_require_explicit_save_and_preserve_letter_spaces(self):
        self.stop_server()
        rows = read_jsonl(self.annotations)
        target = next(row for row in rows if row["text"] == WHOLE_READING_TEXT)
        target["review"] = self.approved_review(target)
        pipeline_fixture.write_rows(self.annotations, rows)
        before = self.annotations.read_bytes()
        envelope = self.suggestion_envelope()
        envelope.update(schema_version=2, policy_version="bilibili-ai-review-v4")
        candidate = "项目用J D K八tomcat九。"
        suggestion = envelope["items"][0]
        suggestion.update(
            id=target["id"], text_sha256=hashlib.sha256(WHOLE_READING_TEXT.encode("utf-8")).hexdigest(),
            decision="edit", acceptable_outputs=[candidate], track="context_reading",
            family="英文单词与字母串",
            reading_audit={
                "status": "needs_reading_annotation", "summary": "Whole reading candidate for human review.",
                "issues": [], "spans": [
                    {"start": start, "end": end, "surface": surface, "occurrence": 1,
                     "mode": mode, "reading": reading, "reason": "Synthetic reading proposal."}
                    for start, end, surface, mode, reading in [
                        (3, 6, "jdk", "letters_en", "J D K"),
                        (6, 7, "8", "number_zh", "八"),
                        (7, 13, "tomcat", "word_en", "tomcat"),
                        (13, 14, "9", "number_zh", "九")]]})
        sidecar = self.write_suggestions(envelope)
        sidecar_bytes = sidecar.read_bytes()
        self.start_server()

        state = self.state()
        self.assertEqual(state["suggestions"], envelope)
        self.assertEqual(state["suggestions_sha256"], hash_file(sidecar))
        self.assertEqual(state["items"], rows)
        self.assertEqual(self.annotations.read_bytes(), before)
        self.assertFalse(Path(self.server.history_path).exists())
        self.assertFalse((self.prepared / "cases.jsonl").exists())

        target = next(row for row in state["items"] if row["id"] == target["id"])
        review = copy.deepcopy(target["review"])
        review.update(acceptable_outputs=suggestion["acceptable_outputs"], track=suggestion["track"],
                      family=suggestion["family"], notes="Human explicitly reviewed this complete reading.")
        status, result, _ = self.request("POST", "api/save", self.payload(state, target, review))
        self.assertEqual(status, 200, result)
        self.assertEqual(result["item"]["review"], review)
        self.assertEqual(result["item"]["review"]["acceptable_outputs"], [candidate])
        self.assertEqual(set(result["item"]["review"]), review_server.REVIEW_KEYS)
        for row in read_jsonl(self.annotations):
            if row["id"] == target["id"]:
                self.assertEqual(row["review"], review)
            else:
                self.assertEqual(row, next(old for old in rows if old["id"] == row["id"]))
        change = next(Path(self.server.history_path).glob("*/change.json"))
        audit = read_json(change)
        self.assertEqual(audit["status"], "committed")
        self.assertEqual(audit["available_ai_suggestion"]["policy_version"], "bilibili-ai-review-v4")
        self.assertEqual(audit["available_ai_suggestion"]["sha256"], hash_file(sidecar))
        self.assertIs(audit["available_ai_suggestion"]["human_reviewed"], False)
        self.assertEqual(change.with_name("before.jsonl").read_bytes(), before)
        self.assertEqual(sidecar.read_bytes(), sidecar_bytes)
        self.assert_fixed_capture_unchanged()

    def test_v3_audit_mutation_requires_restart_before_read_or_save(self):
        self.stop_server()
        envelope = self.reading_suggestion_envelope()
        self.write_suggestions(envelope)
        self.start_server()
        state = self.state()
        target = next(row for row in state["items"] if row["text"] == READING_TEXT)
        envelope["items"][0]["reading_audit"]["spans"][1]["reading"] = "Changed reading proposal."
        self.write_suggestions(envelope)
        status, error, _ = self.request("GET")
        self.assertEqual(status, 409, error)
        self.assertIn("重新启动", error["error"])
        self.assert_rejected_without_writing(self.payload(state, target), expected=409)
        self.assertFalse(Path(self.server.history_path).exists())

    def test_v3_audit_rejects_invalid_structure_bindings_and_unicode_offsets(self):
        self.stop_server()
        before = self.annotations.read_bytes()
        base = self.reading_suggestion_envelope()

        def audit(value):
            return value["items"][0]["reading_audit"]

        mutations = {
            "wrong_schema_pair": lambda x: x.update(schema_version=1),
            "wrong_policy_pair": lambda x: x.update(policy_version="bilibili-ai-review-v2"),
            "manifest_binding": lambda x: x.update(prepared_manifest_sha256="0" * 64),
            "text_binding": lambda x: x["items"][0].update(text_sha256="0" * 64),
            "missing_audit": lambda x: x["items"][0].pop("reading_audit"),
            "unknown_audit_field": lambda x: audit(x).update(confirmed=True),
            "invalid_audit_type": lambda x: x["items"][0].update(reading_audit=[]),
            "unknown_status": lambda x: audit(x).update(status="human_approved"),
            "empty_summary": lambda x: audit(x).update(summary="  "),
            "issues_not_list": lambda x: audit(x).update(issues="unresolved"),
            "blank_issue": lambda x: audit(x).update(issues=[" "]),
            "spans_not_list": lambda x: audit(x).update(spans={}),
            "text_only_with_spans": lambda x: audit(x).update(status="text_only_ok"),
            "decision_without_issue": lambda x: audit(x).update(status="needs_decision"),
            "unknown_span_field": lambda x: audit(x)["spans"][0].update(confirmed=True),
            "missing_span_field": lambda x: audit(x)["spans"][0].pop("reason"),
            "boolean_offset": lambda x: audit(x)["spans"][0].update(start=False),
            "negative_offset": lambda x: audit(x)["spans"][0].update(start=-1),
            "past_text_end": lambda x: audit(x)["spans"][-1].update(end=len(READING_TEXT) + 1),
            "utf16_offset": lambda x: audit(x)["spans"][1].update(start=5, end=7),
            "wrong_occurrence": lambda x: audit(x)["spans"][2].update(occurrence=1),
            "nonexistent_occurrence": lambda x: audit(x)["spans"][2].update(occurrence=3),
            "boolean_occurrence": lambda x: audit(x)["spans"][1].update(occurrence=True),
            "mismatched_surface": lambda x: audit(x)["spans"][1].update(surface="ai"),
            "duplicate_overlap": lambda x: audit(x)["spans"].insert(1, copy.deepcopy(audit(x)["spans"][0])),
            "unsorted_spans": lambda x: audit(x)["spans"].reverse(),
            "unknown_mode": lambda x: audit(x)["spans"][1].update(mode="guessed_ipa"),
            "nontext_mode": lambda x: audit(x)["spans"][1].update(mode=[]),
            "blank_reading": lambda x: audit(x)["spans"][1].update(reading=" "),
            "surrogate_summary": lambda x: audit(x).update(summary="\ud800"),
            "surrogate_issue": lambda x: audit(x).update(issues=["\ud800"]),
            "surrogate_reading": lambda x: audit(x)["spans"][1].update(reading="\ud800"),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                envelope = copy.deepcopy(base)
                mutate(envelope)
                path = self.prepared / "ai-suggestions.json"
                path.write_text(json.dumps(envelope, ensure_ascii=True), encoding="utf-8")
                with self.assertRaisesRegex(ReviewError, "AI 建议"):
                    create_server(self.prepared, port=0)
                self.assertEqual(self.annotations.read_bytes(), before)
                self.assertFalse(Path(self.annotations.with_name("annotations.history")).exists())
        self.write_suggestions(base).write_bytes(b"\xff invalid UTF-8")
        with self.assertRaisesRegex(ReviewError, "AI 建议"):
            create_server(self.prepared, port=0)
        self.assertEqual(self.annotations.read_bytes(), before)
        self.assert_fixed_capture_unchanged()

    def test_suggestion_schema_and_bindings_fail_closed(self):
        self.stop_server()
        before = self.annotations.read_bytes()
        base = self.suggestion_envelope()
        mutations = [lambda x: x.update(prepared_manifest_sha256="0" * 64),
                     lambda x: x["items"][0].update(text_sha256="0" * 64),
                     lambda x: x["items"].append(copy.deepcopy(x["items"][0])),
                     lambda x: x["items"][0].update(id="unknown-id"),
                     lambda x: x["items"][0].update(context_required=0),
                     lambda x: x["items"][0].update(flags="not-a-list"),
                     lambda x: x["items"][0].update(acceptable_outputs=[42]),
                     lambda x: x["items"][0].update(acceptable_outputs=["Changed preservation text"]),
                     lambda x: x.update(human_reviewed=True),
                     lambda x: x.update(policy_version="bilibili-ai-review-v999"),
                     lambda x: x.update(policy_version=["bilibili-ai-review-v2"]),
                     lambda x: x.update(schema_version=2),
                     lambda x: x.update(policy_version="bilibili-ai-review-v4"),
                     lambda x: x.update(schema_version=2, policy_version="bilibili-ai-review-v4"),
                     lambda x: x.update(policy_version="bilibili-ai-review-v5"),
                     lambda x: x.update(schema_version=2, policy_version="bilibili-ai-review-v5"),
                     lambda x: x.update(created_at_utc="not-a-time")]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                envelope = copy.deepcopy(base)
                mutate(envelope)
                self.write_suggestions(envelope)
                with self.assertRaisesRegex(ReviewError, "AI 建议"):
                    create_server(self.prepared, port=0)
                self.assertEqual(self.annotations.read_bytes(), before)
        self.write_suggestions(base).write_bytes(b"\xff invalid UTF-8")
        with self.assertRaisesRegex(ReviewError, "AI 建议"):
            create_server(self.prepared, port=0)
        self.assertEqual(self.annotations.read_bytes(), before)
        self.assert_fixed_capture_unchanged()

    def test_runtime_suggestion_change_requires_restart_without_saving(self):
        self.stop_server()
        envelope = self.suggestion_envelope()
        self.write_suggestions(envelope)
        self.start_server()
        state = self.state()
        envelope["items"][0]["reason"] = "Changed after the browser loaded its suggestion."
        self.write_suggestions(envelope)
        status, error, _ = self.request("GET")
        self.assertEqual(status, 409)
        self.assertIn("重新启动", error["error"])
        self.assert_rejected_without_writing(self.payload(state), expected=409)
        self.assertFalse(Path(self.server.history_path).exists())

    def test_save_changes_only_review_and_preserves_original_bytes_in_history(self):
        # A human editor may use literal Unicode separators and trailing blank lines.
        self.stop_server()
        pipeline_fixture.write_rows(self.annotations, read_jsonl(self.annotations))
        self.annotations.write_bytes(self.annotations.read_bytes() + b"\n\n")
        original_bytes = self.annotations.read_bytes()
        self.start_server()
        state = self.state()
        target = next(item for item in state["items"] if item["text"] == UNICODE_TEXT)
        payload = self.payload(state, target)
        status, result, _ = self.request("POST", "api/save", payload)
        self.assertEqual(status, 200, result)
        self.assertNotEqual(result["version"], state["version"])
        self.assertEqual(result["item"]["review"], payload["review"])
        self.assertEqual({k: v for k, v in result["item"].items() if k != "review"},
                         {k: v for k, v in target.items() if k != "review"})
        saved = {item["id"]: item for item in read_jsonl(self.annotations)}
        self.assertEqual(saved[target["id"]]["text"], UNICODE_TEXT)
        for old in state["items"]:
            new = saved[old["id"]]
            self.assertEqual({k: v for k, v in new.items() if k != "review"},
                             {k: v for k, v in old.items() if k != "review"})
            if old["id"] != target["id"]:
                self.assertEqual(new, old)
        snapshots = list(Path(self.server.history_path).glob("*/before.jsonl"))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].read_bytes(), original_bytes)
        self.assertEqual(read_json(snapshots[0].with_name("change.json"))["status"], "committed")
        self.assert_fixed_capture_unchanged()

    def test_restart_resumes_saved_review_with_a_new_access_token(self):
        state = self.state()
        payload = self.payload(state)
        status, result, _ = self.request("POST", "api/save", payload)
        self.assertEqual(status, 200, result)
        saved_bytes, previous_token = self.annotations.read_bytes(), self.server.token
        self.stop_server()
        self.start_server()
        resumed = self.state()
        self.assertNotEqual(self.server.token, previous_token)
        item = next(item for item in resumed["items"] if item["id"] == payload["id"])
        self.assertEqual(item["review"], payload["review"])
        self.assertEqual(saved_bytes, self.annotations.read_bytes())
        self.assert_fixed_capture_unchanged()

    def test_startup_rejects_a_modified_frozen_review_before_serving_it(self):
        self.stop_server()
        frozen = self.prepared / "review.jsonl"
        frozen.write_bytes(frozen.read_bytes() + b"\n")
        annotation_bytes = self.annotations.read_bytes()
        with self.assertRaises(ReviewError):
            create_server(self.prepared, port=0)
        self.assertEqual(annotation_bytes, self.annotations.read_bytes())

    def test_failed_atomic_replacement_keeps_original_and_audits_failure(self):
        state = self.state()
        original = self.annotations.read_bytes()
        actual_replace = review_server.os.replace

        def fail_only_annotation_replace(source, destination):
            if Path(destination).resolve() == self.annotations.resolve():
                raise OSError("Synthetic disk failure while replacing annotations")
            return actual_replace(source, destination)

        with patch.object(review_server.os, "replace", side_effect=fail_only_annotation_replace):
            self.assert_rejected_without_writing(self.payload(state), expected=500)
        snapshots = list(Path(self.server.history_path).glob("*/before.jsonl"))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].read_bytes(), original)
        self.assertEqual(read_json(snapshots[0].with_name("change.json"))["status"], "failed")
        self.assertEqual(list(self.annotations.parent.glob(".annotations.jsonl.*.tmp")), [])
        self.assertFalse((self.annotations.parent / ".annotations.jsonl.review.lock").exists())

    def test_stale_browser_version_cannot_overwrite_a_saved_decision(self):
        state = self.state()
        payload = self.payload(state)
        status, result, _ = self.request("POST", "api/save", payload)
        self.assertEqual(status, 200, result)
        stale = self.payload(state, state["items"][1])
        self.assert_rejected_without_writing(stale, expected=409)

    def test_external_annotation_edit_is_not_overwritten(self):
        state = self.state()
        external = read_jsonl(self.annotations)
        external[0]["review"]["notes"] = "Changed by a separate editor after the browser loaded."
        pipeline_fixture.write_rows(self.annotations, external)
        external_bytes = self.annotations.read_bytes()
        self.assert_rejected_without_writing(self.payload(state), expected=409)
        self.assertEqual(external_bytes, self.annotations.read_bytes())

    def test_two_clients_with_the_same_version_cannot_both_commit(self):
        state = self.state()
        payloads = [self.payload(state, item) for item in state["items"][:2]]
        start = threading.Barrier(2)

        def save(payload):
            start.wait(timeout=3)
            return self.request("POST", "api/save", payload)[0]

        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(save, payloads))
        self.assertEqual(sorted(statuses), [200, 409])
        self.assertEqual(sum(row["review"]["status"] == "approved"
                             for row in read_jsonl(self.annotations)), 1)
        self.assert_fixed_capture_unchanged()

    def test_pending_drafts_allow_incomplete_references_without_implying_approval(self):
        state = self.state()
        item = state["items"][0]
        draft = copy.deepcopy(item["review"])
        draft["notes"] = "Need to discuss the intended reading.\nNo reference approved yet."
        self.assertEqual(draft["status"], "pending")
        self.assertEqual(draft["reviewer"], "")
        self.assertEqual(draft["acceptable_outputs"], [])
        status, result, _ = self.request("POST", "api/save", self.payload(state, item, draft))
        self.assertEqual(status, 200, result)
        self.assertEqual(result["item"]["review"], draft)
        self.assertFalse((self.prepared / "inputs.jsonl").exists())
        self.assertFalse((self.prepared / "cases.jsonl").exists())

    def test_invalid_approvals_and_unknown_review_fields_are_rejected(self):
        state = self.state()
        mutations = [("reviewer", "  "), ("reviewer", 123),
                     ("acceptable_outputs", []), ("acceptable_outputs", [""]),
                     ("acceptable_outputs", [42]), ("acceptable_outputs", "not a list"),
                     ("track", "unknown"), ("family", ""), ("context_required", True),
                     ("context_required", None), ("context_required", 0),
                     ("status", "unknown"), ("unexpected_gold", "not allowed")]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                payload = self.payload(state)
                payload["review"][field] = value
                self.assert_rejected_without_writing(payload)
        wrong_reference = self.payload(state)
        wrong_reference["review"]["acceptable_outputs"] = ["This changes preservation text."]
        self.assert_rejected_without_writing(wrong_reference)
        missing_field = self.payload(state)
        del missing_field["review"]["family"]
        self.assert_rejected_without_writing(missing_field)

    def test_unknown_ids_and_attempts_to_change_source_text_are_rejected(self):
        state = self.state()
        unknown = self.payload(state)
        unknown["id"] = "not-in-the-frozen-batch"
        self.assert_rejected_without_writing(unknown)
        for field, value in (("text", "Injected reading answer."), ("source", {"repo": "changed"}),
                             ("acceptable_outputs", ["Gold outside review."])):
            with self.subTest(field=field):
                payload = self.payload(state)
                payload[field] = value
                self.assert_rejected_without_writing(payload)

    def test_access_requires_matching_host_origin_and_both_tokens(self):
        state = self.state()
        for headers in ({"Host": "attacker.example"},
                        {"Host": f"localhost:{self.server.server_port}"},
                        {"Origin": "https://attacker.example"},
                        {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                self.assert_rejected_without_writing(self.payload(state), expected=403, headers=headers)
                status, _, _ = self.request("GET", headers=headers)
                self.assertEqual(status, 403)
        wrong_body_token = self.payload(state)
        wrong_body_token["token"] = "incorrect"
        self.assert_rejected_without_writing(wrong_body_token, expected=403)
        for path in ("/api/state", "/incorrect/api/state"):
            with self.subTest(path=path):
                status, _, _ = self.request("GET", path=path)
                self.assertIn(status, {403, 404})
        before = self.annotations.read_bytes()
        status, _, _ = self.request("POST", "api/save", self.payload(state), path="/incorrect/api/save")
        self.assertIn(status, {403, 404})
        self.assertEqual(before, self.annotations.read_bytes())


if __name__ == "__main__":
    unittest.main()
