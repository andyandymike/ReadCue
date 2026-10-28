"""Portable research bundle boundaries, using synthetic files and no model runtime."""
import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "readcue_research_bundle_builder", PROJECT / "tools/build_model_research_bundle.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ModelResearchBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def make_bundle(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        body = b"Synthetic local research candidate.\n"
        (bundle / "README.md").write_bytes(body)
        manifest = {
            "schema_version": 1,
            "base": {"repo": "Qwen/Qwen3-0.6B",
                     "revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
            "files": {"README.md": {"sha256": hashlib.sha256(body).hexdigest(),
                                    "bytes": len(body)}},
        }
        self.write_manifest(bundle, manifest)
        return bundle, manifest

    @staticmethod
    def write_manifest(bundle, manifest):
        (bundle / "bundle-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")

    def portable_namespace(self):
        source = builder.portable_source(PROJECT)
        namespace = {"__name__": "portable_bundle_test",
                     "__file__": str(self.root / "inference.py")}
        original_import = builtins.__import__

        def no_model_runtime(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "peft"}:
                raise AssertionError("Input and integrity helpers must not import " + name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_model_runtime):
            exec(compile(source, "<portable-inference>", "exec"), namespace)
        return source, namespace

    def test_function_extraction_keeps_source_without_executing_it(self):
        source = self.root / "source.py"
        text = (
            "raise RuntimeError('must never execute source during extraction')\n\n"
            "def selected(value):\n"
            "    # This comment and spacing belong to the frozen implementation.\n"
            "    return value  +  1\n\n"
            "def excluded(value):\n"
            "    return 'do not copy this function'\n"
        )
        source.write_text(text, encoding="utf-8")
        extracted = builder.extract_functions(source, ["selected"])
        selected = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef))
        self.assertIn(ast.get_source_segment(text, selected), extracted)
        self.assertNotIn("must never execute", extracted)
        self.assertNotIn("def excluded", extracted)
        namespace = {}
        exec(compile(extracted, "<copied-helper>", "exec"), namespace)
        self.assertEqual(namespace["selected"](9), 10)

    def test_extraction_rejects_missing_functions(self):
        source = self.root / "source.py"
        source.write_text("def present():\n    return 1\n", encoding="utf-8")
        with self.assertRaises((ValueError, KeyError)):
            builder.extract_functions(source, ["missing"])

    def test_frozen_loading_helper_is_copied_verbatim(self):
        source = PROJECT / "scripts/run_zh_baseline.py"
        text = source.read_text(encoding="utf-8")
        name = "load_local_tokenizer_and_config"
        node = next(item for item in ast.parse(text).body
                    if isinstance(item, ast.FunctionDef) and item.name == name)
        extracted = builder.extract_functions(source, [name])
        self.assertIn(ast.get_source_segment(text, node), extracted)
        self.assertIn("tokenizer.get_vocab() != expected_vocab", extracted)
        self.assertIn("actual.special != entry['special']", extracted)
        self.assertIn("Only the audited default RoPE migration is supported", extracted)

    def test_checked_relative_accepts_a_plain_contained_file(self):
        nested = self.root / "nested"
        nested.mkdir()
        expected = nested / "asset.json"
        expected.write_text("{}", encoding="utf-8")
        self.assertEqual(builder.checked_relative(self.root, "nested/asset.json"), expected)

    def test_checked_relative_rejects_absolute_and_parent_paths(self):
        outside = self.root.parent / "outside.json"
        for relative in ("../outside.json", "nested/../../outside.json", str(outside),
                         "/absolute.json", "C:/outside.json", "C:\\outside.json"):
            with self.subTest(relative=relative):
                with self.assertRaises((ValueError, OSError)):
                    builder.checked_relative(self.root, relative)

    def test_checked_relative_rejects_links_inside_the_tree(self):
        target = self.root / "actual"
        target.mkdir()
        (target / "asset.json").write_text("{}", encoding="utf-8")
        link = self.root / "linked"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest("This host cannot create test symlinks: " + str(error))
        with self.assertRaises((ValueError, OSError)):
            builder.checked_relative(self.root, "linked/asset.json")

    def test_existing_output_is_refused_without_touching_its_contents(self):
        output = self.root / "already-present"
        output.mkdir()
        sentinel = output / "keep.bin"
        sentinel.write_bytes(b"existing candidate must stay byte-identical")
        before = sentinel.read_bytes()
        with self.assertRaises(FileExistsError):
            builder.build_bundle(self.root / "missing-project", self.root / "missing-run",
                                 self.root / "missing-base", output)
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertEqual({item.name for item in output.iterdir()}, {"keep.bin"})

    def test_bundle_verifies_registered_payload_and_allows_separate_checks(self):
        bundle, manifest = self.make_bundle()
        checks = bundle / "checks"
        checks.mkdir()
        (checks / "cpu-check.json").write_text('{"synthetic": true}\n', encoding="utf-8")
        self.assertEqual(builder.verify_bundle(bundle), manifest)

    def test_bundle_rejects_hash_and_size_tampering(self):
        bundle, manifest = self.make_bundle()
        path = bundle / "README.md"
        original = path.read_bytes()
        path.write_bytes(b"X" + original[1:])
        with self.assertRaises(ValueError):
            builder.verify_bundle(bundle)
        path.write_bytes(original)
        manifest["files"]["README.md"]["bytes"] += 1
        self.write_manifest(bundle, manifest)
        with self.assertRaises(ValueError):
            builder.verify_bundle(bundle)

    def test_bundle_rejects_unregistered_payload(self):
        bundle, _ = self.make_bundle()
        (bundle / "train.jsonl").write_text('{"target":"must not be packaged"}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            builder.verify_bundle(bundle)

    def test_bundle_manifest_cannot_reach_outside_its_root(self):
        bundle, manifest = self.make_bundle()
        secret = self.root / "outside.bin"
        secret.write_bytes(b"not a bundle asset")
        manifest["files"]["../outside.bin"] = {
            "sha256": hashlib.sha256(secret.read_bytes()).hexdigest(),
            "bytes": secret.stat().st_size,
        }
        self.write_manifest(bundle, manifest)
        with self.assertRaises((ValueError, OSError)):
            builder.verify_bundle(bundle)
        self.assertEqual(secret.read_bytes(), b"not a bundle asset")

    def test_bundle_rejects_a_moving_base_revision(self):
        bundle, manifest = self.make_bundle()
        manifest["base"]["revision"] = "main"
        self.write_manifest(bundle, manifest)
        with self.assertRaises(ValueError):
            builder.verify_bundle(bundle)

    def test_portable_input_contract_rejects_gold_and_duplicate_ids(self):
        _, namespace = self.portable_namespace()
        parse = namespace["parse_inputs"]
        for extra in ("target", "acceptable_outputs", "track", "targets", "rationale"):
            with self.subTest(extra=extra):
                body = json.dumps({"id": "example-1", "text": "原句12元。", extra: "gold"})
                with self.assertRaises(ValueError):
                    parse(body)
        duplicate = '{"id":"same","text":"甲"}\n{"id":"same","text":"乙"}\n'
        with self.assertRaises(ValueError):
            parse(duplicate)
        for body in ("", '{"id":" ","text":"句子"}', '{"id":"x","text":" "}'):
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    parse(body)

    def test_portable_input_contract_preserves_unicode_separators_and_text(self):
        _, namespace = self.portable_namespace()
        row = {"id": "only-id", "text": "保留\u0085\u2028\u2029以及  12元。"}
        body = json.dumps(row, ensure_ascii=False) + "\n"
        self.assertEqual(namespace["parse_inputs"](body), [row])

    def test_portable_contains_exact_frozen_loading_and_prompt_helpers(self):
        portable, _ = self.portable_namespace()
        for relative, name in (
                ("scripts/run_zh_baseline.py", "load_local_tokenizer_and_config"),
                ("src/readcue/model_experiment.py", "prompt_tokens"),
                ("src/readcue/model_experiment.py", "_predict"),
                ("src/readcue/inference_contract.py", "parse_inputs")):
            with self.subTest(helper=name):
                text = (PROJECT / relative).read_text(encoding="utf-8")
                node = next(item for item in ast.parse(text).body
                            if isinstance(item, ast.FunctionDef) and item.name == name)
                self.assertIn(ast.get_source_segment(text, node), portable)

    def test_portable_prompt_uses_only_fixed_system_prompt_and_original_text(self):
        _, namespace = self.portable_namespace()
        calls = []

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return "fixed rendered prompt"

            def encode(self, text, **kwargs):
                calls.append((text, kwargs))
                return [10, 20]

        text = "把 target 和参考当原文保留：12元。"
        prompt, ids = namespace["prompt_tokens"](Tokenizer(), {"system_prompt": "固定政策"}, text)
        self.assertEqual((prompt, ids), ("fixed rendered prompt", [10, 20]))
        self.assertEqual(calls[0], ([{"role": "system", "content": "固定政策"},
                                    {"role": "user", "content": text}],
                                   {"tokenize": False, "add_generation_prompt": True,
                                    "enable_thinking": False}))
        self.assertEqual(calls[1], ("fixed rendered prompt", {"add_special_tokens": False}))

    def test_portable_output_cannot_traverse_from_checks_into_payload(self):
        _, namespace = self.portable_namespace()
        (self.root / "checks").mkdir()
        (self.root / "adapter").mkdir()
        output = self.root / "checks" / ".." / "adapter" / "new-results"
        with patch.dict(namespace["os"].environ, {}):
            with self.assertRaisesRegex(ValueError, "under checks"):
                namespace["main"](["check", "--base-dir", str(self.root / "base"),
                                   "--output", str(output)])
        self.assertFalse((self.root / "adapter/new-results").exists())

    def test_portable_output_rejects_symlink_alias(self):
        _, namespace = self.portable_namespace()
        (self.root / "adapter").mkdir()
        link = self.root / "alias"
        try:
            link.symlink_to(self.root / "adapter", target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest("This host cannot create test symlinks: " + str(error))
        with patch.dict(namespace["os"].environ, {}):
            with self.assertRaisesRegex(ValueError, "links or reparse"):
                namespace["main"](["check", "--base-dir", str(self.root / "base"),
                                   "--output", str(link / "new-results")])
        self.assertFalse((self.root / "adapter/new-results").exists())


if __name__ == "__main__":
    unittest.main()
