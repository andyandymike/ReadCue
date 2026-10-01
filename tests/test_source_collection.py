"""Offline checks for source preservation and structural Markdown exclusion."""
import importlib.util
import tempfile
from collections import Counter
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools/collect_zh_mdn.py"
SPEC = importlib.util.spec_from_file_location("collect_zh_mdn", SOURCE)
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


class SourceCollectionTests(unittest.TestCase):
    def test_inline_code_literals_survive_presentation_cleanup(self):
        text, reason = COLLECTOR.display_prose("请保留 `__proto__`、`2**3` 和 `[1]` 这些原始标识。")
        self.assertIsNone(reason)
        self.assertEqual(text, "请保留 __proto__、2**3 和 [1] 这些原始标识。")

    def test_escaped_punctuation_and_regex_stay_literal(self):
        text, reason = COLLECTOR.display_prose(r"这里是 \*DAV 与 \[1]，正则 `\d` 和 `\1` 应保留。")
        self.assertIsNone(reason)
        self.assertEqual(text, r"这里是 *DAV 与 [1]，正则 \d 和 \1 应保留。")
        self.assertEqual(COLLECTOR.display_prose("这里的数据*隐蔽地*发生改变。")[1],
                         "unresolved_emphasis_or_literal")

    def test_structural_content_and_continuations_are_not_prose(self):
        markdown = """---
title: 这里是元数据，不是正文。
---

## 标题

这是带有版本 3.10 的完整正文句子。

```text
这个代码围栏里的句子不能被采入。
```

> 这个提示块里的句子不能被采入。
这个提示块的惰性续行也不能被采入。

- 这个列表里的句子不能被采入。
  列表继续描述的这一行也不能被采入。

  空行之后的列表缩进段落也不能被采入。

| 表头 |
| --- |
| 表格中的内容也不能作为普通正文。 |

这是一个含有 {{HTTPStatus(404)}} 的宏段落。

最后这句正文完整保留 1.5 GB 和 HTTP/2。
"""
        accepted = [COLLECTOR.display_prose(block)[0]
                    for _, _, _, block in COLLECTOR.prose_blocks(markdown)]
        self.assertEqual([text for text in accepted if text], [
            "这是带有版本 3.10 的完整正文句子。", "最后这句正文完整保留 1.5 GB 和 HTTP/2。"])

    def test_presentation_edits_preserve_tokens_and_exact_raw_paragraph(self):
        body = "## 说明\n\n服务器返回 **`404`**，请查阅[这份说明](/docs/status/404)。\n".encode()
        document = {"id": "document", "source_url": "https://example.test/fixed/document.md",
                    "sha256": COLLECTOR.digest(body), "revision": "a" * 40, "group": "fixture"}
        with tempfile.TemporaryDirectory() as temporary:
            rows = COLLECTOR.extract_document(body, document, Path(temporary), {}, Counter())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["text"], "服务器返回 404，请查阅这份说明。")
        self.assertEqual(row["raw_paragraph"], body.decode().splitlines()[2])
        self.assertEqual((row["line_start"], row["line_end"]), (3, 3))
        self.assertFalse(row["human_reviewed"])
        self.assertEqual(row["annotation_status"], "unlabeled")
        self.assertNotIn("acceptable_outputs", row)

    def test_duplicate_origin_is_retained_without_changing_text(self):
        body = "这里是包含数字 404 的完整中文正文。\n".encode()
        document = {"id": "first", "source_url": "https://example.test/fixed/first.md",
                    "sha256": COLLECTOR.digest(body), "revision": "a" * 40, "group": "fixture"}
        seen, rejected = {}, Counter()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = COLLECTOR.extract_document(body, document, output, seen, rejected)
            second = COLLECTOR.extract_document(body, {**document, "id": "second"}, output, seen, rejected)
            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])
            self.assertIn('"document_id": "second"', (output / "duplicates.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(rejected["exact_duplicate"], 1)

    def test_closing_parentheses_and_quotes_stay_with_sentence(self):
        body = '这里包含一个完整的解释（服务器允许访问文件。）后面是独立而完整的下一句话。'.encode()
        document = {"id": "document", "source_url": "https://example.test/fixed/document.md",
                    "sha256": COLLECTOR.digest(body), "revision": "a" * 40, "group": "fixture"}
        with tempfile.TemporaryDirectory() as temporary:
            rows = COLLECTOR.extract_document(body, document, Path(temporary), {}, Counter())
        self.assertEqual([row["text"] for row in rows], [
            "这里包含一个完整的解释（服务器允许访问文件。）", "后面是独立而完整的下一句话。"])


if __name__ == "__main__":
    unittest.main()
