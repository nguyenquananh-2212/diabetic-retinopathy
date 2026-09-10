"""Focused regression tests for sampler, progress and the RAR upload workflow."""

import ast
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Source" / "training"))
sys.path.insert(0, str(ROOT / "Source" / "preprocessing"))
import train as training
import data as training_data
import package_for_kaggle as packaging


def notebook_cells():
    return json.loads((ROOT / "Notebooks" / "kaggle_train_448.ipynb").read_text(encoding="utf-8"))["cells"]


def notebook_functions(marker):
    source = next("".join(c["source"]) for c in notebook_cells() if marker in "".join(c["source"]))
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {"DEPTH": 9}
    exec(compile(tree, "notebook_helpers", "exec"), namespace)
    return namespace


class TrainingUploadTests(unittest.TestCase):
    def test_notebook_compiles_and_is_clean(self):
        for index, cell in enumerate(notebook_cells()):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell_{index}", "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_config_defaults_and_cli(self):
        self.assertEqual(training.TrainConfig().sampler_strength, 0.5)
        self.assertEqual(training.TrainConfig().loss_weight_strength, 0.0)
        self.assertTrue(training.TrainConfig().progress)
        with patch.object(sys, "argv", ["train.py"]):
            args = training.parse_args()
        self.assertEqual(args.sampler_strength, 0.5)
        self.assertTrue(args.progress)
        with patch.object(sys, "argv", ["train.py", "--no-progress", "--sampler-strength", "1"]):
            args = training.parse_args()
        self.assertFalse(args.progress)
        self.assertEqual(args.sampler_strength, 1.0)
        with patch.object(sys, "argv", ["package_for_kaggle.py"]):
            self.assertEqual(packaging.parse_args().format, "rar")

    def test_notebook_explicit_defaults_for_both_models(self):
        source = next("".join(c["source"]) for c in notebook_cells() if "SAMPLER_STRENGTH =" in "".join(c["source"]))
        ns = {"os": __import__("os"), "native_bf16": False, "DATA_DIR": ROOT,
              "WORK": ROOT, "IMAGE_ROOT": ROOT}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, "notebook_config", "exec"), ns)
            ns["train"] = lambda cfg, *args: cfg
            for backbone in ("convnext_tiny", "swin_t"):
                cfg = ns["run"](backbone)
                self.assertEqual(cfg.sampler_strength, 0.5)
                self.assertEqual(cfg.loss_weight_strength, 0.0)
                self.assertTrue(cfg.progress)
                self.assertEqual(cfg.batch_size * cfg.accumulate, 36)
            self.assertEqual(ns["run"]("convnext_tiny", sampler_strength=1.0).sampler_strength, 1.0)

    def test_sampler_uses_square_root_weights(self):
        samples = [SimpleNamespace(label=0)] * 4 + [SimpleNamespace(label=1)]
        sampler = training_data.balanced_sampler(samples)
        self.assertAlmostEqual(float(sampler.weights[-1] / sampler.weights[0]), 2.0)
        self.assertEqual(sampler.num_samples, len(samples))
        self.assertTrue(sampler.replacement)

    def test_live_progress_and_disable(self):
        stream = io.StringIO()

        def factory(*args, **kwargs):
            kwargs.update(file=stream, mininterval=0, miniters=1)
            return tqdm(*args, **kwargs)

        with patch.object(training, "tqdm", factory):
            bar = training.progress(range(3), "convnext_tiny train 1/2", total=3)
            for _ in bar:
                bar.set_postfix(loss="0.1234", lr="1e-4", refresh=False)
            self.assertEqual(bar.n, 3)
            self.assertEqual(bar.unit, "batch")
            self.assertIn("3/3", stream.getvalue())
            self.assertIn("convnext_tiny train 1/2", stream.getvalue())
            self.assertIn("loss=0.1234", stream.getvalue())
        values = [1, 2]
        self.assertIs(training.progress(values, "off", enable=False), values)
        with patch.object(training, "tqdm", None):
            self.assertIs(training.progress(values, "missing"), values)

    def test_two_epoch_cpu_training_progress(self):
        samples = TensorDataset(torch.randn(20, 3, 8, 8), torch.arange(20) % 5)
        loader = DataLoader(samples, batch_size=5)
        stages = []

        def progress_spy(iterable, desc, total=None, enable=True):
            stages.append((desc, total, enable))
            return iterable

        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 5))
        with tempfile.TemporaryDirectory(prefix="fundus-train-test-") as tmp, \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(training, "build_loaders", return_value=(loader, loader, loader, torch.ones(5))), \
             patch.object(training, "build_model", return_value=model), \
             patch.object(training, "progress", side_effect=progress_spy), \
             contextlib.redirect_stdout(io.StringIO()):
            summary = training.train(
                training.TrainConfig(epochs=2, amp="off", accumulate=2, workers=0),
                Path("unused_manifest.csv"), Path("unused_splits.csv"), Path(tmp),
            )
            self.assertEqual(len(summary["history"]), 2)
            self.assertEqual(summary["config"]["sampler_strength"], 0.5)
        self.assertIn(("convnext_tiny train 1/2", 4, True), stages)
        self.assertIn(("convnext_tiny train 2/2", 4, True), stages)
        self.assertTrue(any("TTA x8" in label for label, _, _ in stages))

    def test_rar_roundtrip_cache_and_failed_rebuild(self):
        ns = notebook_functions("def rar_extractor():")
        tool = ns["rar_extractor"]()
        with tempfile.TemporaryDirectory(prefix="fundus-rar-test-") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            payload = "sampler_strength = 0.5\n# tiếng Việt\n".encode("utf-8")
            (source / "train.py").write_bytes(payload)
            target = root / "dr448_code.rar"
            with contextlib.redirect_stdout(io.StringIO()):
                packaging.write_rar(target, source, ["train.py"], "Source/training", "test", 1)
                destination = ns["extract_rar"](target, root / "cache", tool)
            self.assertEqual((destination / "Source/training/train.py").read_bytes(), payload)
            with patch.object(subprocess, "run", side_effect=AssertionError("cache should not extract again")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ns["extract_rar"](target, root / "cache", tool), destination)
            previous = target.read_bytes()
            failed = subprocess.CompletedProcess([], 2, "", "simulated failure")
            with patch.object(packaging.subprocess, "run", return_value=failed):
                with self.assertRaises(SystemExit):
                    packaging.write_rar(target, source, ["train.py"], "Source/training", "test", 1)
            self.assertEqual(target.read_bytes(), previous)
            self.assertEqual(list(root.glob(".rar-build-*")), [])

    def test_reject_unsafe_archive_path(self):
        ns = notebook_functions("def rar_extractor():")
        tool = ns["rar_extractor"]()
        with tempfile.TemporaryDirectory(prefix="fundus-unsafe-test-") as tmp:
            root = Path(tmp)
            archive = root / "dr448_unsafe.rar"
            # bsdtar detects content, regardless of extension.  No RAR writer
            # should create traversal entries, so use ZIP only for this test.
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", "must not extract")
            with self.assertRaisesRegex(RuntimeError, "không hợp lệ"):
                ns["extract_rar"](archive, root / "cache", tool)
            self.assertFalse((root / "outside.txt").exists())

    def test_discovery_reads_extracted_branches(self):
        ns = notebook_functions("def find(pattern")
        with tempfile.TemporaryDirectory(prefix="fundus-discovery-test-") as tmp:
            root = Path(tmp)
            tables = root / "dr448_tables/img448"
            code = root / "dr448_code/Source/training"
            tables.mkdir(parents=True)
            code.mkdir(parents=True)
            for path in (tables / "manifest.csv", tables / "splits.csv", code / "train.py", code / "data.py"):
                path.touch()
            ns["SEARCH_ROOTS"] = [root / "dr448_tables", root / "dr448_code"]
            self.assertEqual(ns["find"]("manifest.csv", "splits.csv"), [tables / "manifest.csv"])
            self.assertEqual(ns["find"]("train.py", "data.py"), [code / "train.py"])


if __name__ == "__main__":
    unittest.main()
