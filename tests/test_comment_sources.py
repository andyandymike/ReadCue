"""Synthetic, offline checks for faithful and bounded comment imports."""
import copy
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import io
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from readcue.comment_sources import import_comments


HEADER = ["bv_id", "oid", "level", "root_rpid", "rpid", "parent_rpid", "uid", "uname",
          "user_level", "ctime", "like", "message"]


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(HEADER)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def comment(text="哈", **changes):
    row = dict(zip(HEADER, ["BV15ysoeqEcm", "113208627172171", "1", "123", "123", "0",
                           "987", "fixture author", "5", "1750000000", "0", text]))
    row.update(changes)
    return [row[key] for key in HEADER]


class CommentSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "capture"
        self.root.mkdir()
        self.config = {"source_id": "fixture", "repo": "example/comments", "revision": "a" * 40,
                       "capture_date": "2026-09-29", "license": {"status": "unresolved", "note": "Fixture only"},
                       "files": [], "evidence_files": []}

    def add_file(self, body, name="comments.csv", evidence=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        record = {"path": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                  "git_blob_sha": hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()}
        self.config["evidence_files" if evidence else "files"].append(record)
        return path

    def test_text_bytes_semantics_and_source_spans_survive(self):
        texts = ["哈", "  @某人\t好耶😀\r\n下一行\n尾部  ", '大小写 A_b，逗号,"引号"']
        body = csv_bytes([comment(text, rpid=str(123 + i)) for i, text in enumerate(texts)])
        path = self.add_file(body, "data/comments.csv")
        self.add_file(b"Fixture license evidence\n", "README.md", evidence=True)
        result = import_comments(self.root, self.config)
        self.assertEqual(path.read_bytes(), body)
        self.assertEqual([row["text"] for row in result["records"]], texts)
        self.assertEqual(result["rejected"], [])
        self.assertEqual([row["role"] for row in result["sources"]], ["data", "evidence"])
        row = result["records"][1]
        self.assertEqual(row["id"], "fixture:BV15ysoeqEcm:124")
        self.assertEqual(row["parent_id"], "0")
        self.assertEqual(row["source"]["record_number"], 3)
        self.assertEqual((row["source"]["line_start"], row["source"]["line_end"]), (3, 5))
        self.assertTrue(row["comment_time"].endswith("+08:00"))
        self.assertNotIn("uid", row)
        self.assertNotIn("uname", row)
        self.assertNotIn("fixture author", repr(row))
        self.assertNotIn("raw_row", row)
        self.assertEqual(row["quality_flags"], ["multiline_text", "boundary_whitespace"])

    def test_all_file_and_evidence_hashes_checked_before_parsing(self):
        self.add_file(b'broken,"unclosed')
        evidence = self.add_file(b"original", "README.md", evidence=True)
        evidence.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "bytes/hash mismatch: README.md"):
            import_comments(self.root, self.config)

    def test_each_hash_and_size_are_required(self):
        self.add_file(csv_bytes([comment()]))
        for field, invalid in (("sha256", "b" * 64), ("git_blob_sha", "c" * 40), ("bytes", 0)):
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config["files"][0][field] = invalid
                with self.assertRaisesRegex(ValueError, "bytes/hash mismatch"):
                    import_comments(self.root, config)

    def test_traversal_absolute_paths_and_ads_are_rejected(self):
        self.add_file(csv_bytes([comment()]))
        for path in ("../outside.csv", "/outside.csv", "C:/outside.csv", "nested/../comments.csv",
                     "comments.csv:stream", "nested\\comments.csv"):
            with self.subTest(path=path):
                config = copy.deepcopy(self.config)
                config["files"][0]["path"] = path
                with self.assertRaisesRegex(ValueError, "source-relative path"):
                    import_comments(self.root, config)

    def test_symlink_file_and_parent_are_rejected(self):
        self.add_file(csv_bytes([comment()]))
        original_lstat = Path.lstat
        # Simulate lstat metadata: Windows need not grant symlink-creation rights.
        for relative, link_part in (("comments.csv", "comments.csv"), ("alias/comments.csv", "alias")):
            with self.subTest(relative=relative):
                config = copy.deepcopy(self.config)
                config["files"][0]["path"] = relative
                linked_path = self.root.resolve() / link_part

                def linked_lstat(path):
                    if path == linked_path:
                        return SimpleNamespace(st_mode=stat.S_IFLNK)
                    return original_lstat(path)

                with patch.object(Path, "lstat", linked_lstat):
                    with self.assertRaisesRegex(ValueError, "links or reparse"):
                        import_comments(self.root, config)

    def test_bad_business_rows_are_kept_in_rejection_evidence(self):
        bad_rows = [comment(rpid="1.23E+11"), comment(oid="0"), comment(bv_id="not-a-video"),
                    comment(root_rpid="-1"), comment(parent_rpid="1.0"), comment(ctime="2025-01-01"),
                    comment(ctime="1"), comment(ctime="9999999999"), comment(text=" \t "), ["bad", "columns"]]
        self.add_file(csv_bytes([comment(), *bad_rows, comment(root_rpid="0")]))
        result = import_comments(self.root, self.config)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(len(result["rejected"]), len(bad_rows))
        self.assertEqual([row["raw_row"] for row in result["rejected"]], bad_rows)
        self.assertEqual(result["rejected"][0]["reasons"], ["scientific_notation:rpid"])
        self.assertEqual(result["rejected"][-1]["reasons"], ["column_count_mismatch"])
        self.assertEqual(result["rejected"][0]["source"]["record_number"], 3)

    def test_capture_day_uses_china_time_and_max_date_only_narrows(self):
        zone = timezone(timedelta(hours=8))
        timestamp = lambda year, month, day: str(int(datetime(year, month, day, tzinfo=zone).timestamp()))
        self.add_file(csv_bytes([comment(ctime=timestamp(2009, 1, 1)),
                                 comment(ctime=str(int(timestamp(2009, 1, 1)) - 1)),
                                 comment(ctime=str(int(timestamp(2026, 9, 30)) - 1)),
                                 comment(ctime=timestamp(2026, 9, 30))]))
        result = import_comments(self.root, self.config)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(len(result["rejected"]), 2)
        self.assertEqual(result["records"][-1]["comment_time"], "2026-09-29T23:59:59+08:00")
        config = {**self.config, "max_date": "2026-09-28"}
        self.assertEqual(len(import_comments(self.root, config)["records"]), 1)
        with self.assertRaisesRegex(ValueError, "max_date"):
            import_comments(self.root, {**self.config, "max_date": "2026-09-30"})

    def test_duplicate_comment_ids_remain_for_relationship_audit(self):
        self.add_file(csv_bytes([comment(), comment("另一个文本")]))
        records = import_comments(self.root, self.config)["records"]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["id"], records[1]["id"])

    def test_broken_csv_and_utf8_fail_the_whole_import(self):
        for body in (csv_bytes([comment()]) + b'"unclosed,field', csv_bytes([comment()]) + b'\xff'):
            with self.subTest(body_suffix=body[-5:]):
                self.config["files"] = []
                self.add_file(body)
                with self.assertRaisesRegex(ValueError, "Unrecoverable CSV|Invalid UTF-8"):
                    import_comments(self.root, self.config)

    def test_parse_uses_the_verified_bytes_without_a_second_read(self):
        original = csv_bytes([comment("哈😀")])
        self.add_file(original)
        original_reader = Path.read_bytes
        reads = []

        def read_once(path):
            reads.append(path)
            if len(reads) > 1:
                raise AssertionError("Source was read again after verification")
            return original_reader(path)

        with patch.object(Path, "read_bytes", read_once):
            result = import_comments(self.root, self.config)
        self.assertEqual(result["records"][0]["text"], "哈😀")
        self.assertEqual(len(reads), 1)


if __name__ == "__main__":
    unittest.main()
