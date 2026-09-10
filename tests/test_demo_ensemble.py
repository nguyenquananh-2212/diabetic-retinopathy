"""Regression checks for saved ensemble settings and single-image inference."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Source" / "inference"))
import demo


class FakeGrader:
    probabilities = demo.Grader.probabilities

    def __init__(self, name, logits):
        self.name, self.head, self.size = name, "softmax", 2
        self.logits = np.asarray(logits, dtype=np.float32).reshape(1, 5)
        self.calls = []

    def forward(self, batch, tta=False):
        self.calls.append(tta)
        return self.logits, np.array([[1.0, 2.0]], dtype=np.float32)


class DemoEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.graders = [FakeGrader("convnext_tiny_softmax", [2, 1, 0, -1, -2]),
                        FakeGrader("swin_t_softmax", [-1, 0, 1, 2, 0])]
        names = [g.name for g in self.graders]
        self.config = {"runs": names, "tta": True,
                       "temperatures": dict(zip(names, [0.9, 1.2])),
                       "weights": dict(zip(names, [0.6, 0.4])),
                       "thresholds": [0.7, 1.4, 2.4, 3.3]}

    def load(self, config=None, graders=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.json"
            path.write_text(json.dumps(self.config if config is None else config), encoding="utf-8")
            return demo.Ensemble.from_json(self.graders if graders is None else graders, path)

    def test_saved_tta_temperature_and_probability_blend(self):
        ensemble = self.load(graders=self.graders[::-1])
        self.assertEqual(ensemble.names, self.config["runs"])
        with patch.object(demo, "to_tensor", return_value=torch.zeros(3, 2, 2)):
            blended, per_model, features = ensemble.run(np.zeros((2, 2, 3), np.uint8))
        expected = sum(self.config["weights"][g.name] * demo.M.softmax_probabilities(
            g.logits / self.config["temperatures"][g.name])[0] for g in self.graders)
        np.testing.assert_allclose(blended, expected, atol=1e-7)
        self.assertEqual(set(per_model), set(self.config["runs"]))
        self.assertEqual(set(features), set(self.config["runs"]))
        self.assertTrue(all(g.calls == [True] for g in self.graders))

    def test_tta_defaults_and_conflicts(self):
        ensemble = self.load()
        self.assertTrue(ensemble.resolve_tta())
        with self.assertRaisesRegex(ValueError, "mâu thuẫn"):
            ensemble.run(np.zeros((2, 2, 3)), tta=False)
        self.config["tta"] = False
        self.assertFalse(self.load().resolve_tta())
        with self.assertRaises(ValueError):
            self.load().resolve_tta(True)
        plain = demo.Ensemble(self.graders)
        self.assertFalse(plain.resolve_tta())
        self.assertTrue(plain.resolve_tta(True))

    def test_ordinal_decoder_and_argmax_fallback(self):
        probabilities = np.random.default_rng(7).dirichlet(np.ones(5), size=500)
        ensemble = self.load()
        expected = demo.M.apply_thresholds(demo.M.expected_grade(probabilities), ensemble.thresholds)
        np.testing.assert_array_equal([ensemble.decode(p) for p in probabilities], expected)
        plain = demo.Ensemble(self.graders)
        np.testing.assert_array_equal([plain.decode(p) for p in probabilities], probabilities.argmax(1))

    def test_reject_missing_extra_and_duplicate_models(self):
        extra = FakeGrader("extra_softmax", [0, 0, 0, 0, 0])
        for graders in (self.graders[:1], self.graders + [extra], self.graders * 2):
            with self.subTest(names=[g.name for g in graders]), self.assertRaises(SystemExit):
                self.load(graders=graders)

    def test_reject_invalid_calibration(self):
        name = self.graders[0].name
        cases = [("temperatures", {name: 1.0}),
                 ("weights", {name: 1.0}), ("tta", "true"),
                 ("runs", self.config["runs"] * 2),
                 ("thresholds", [1, 2, 3]), ("thresholds", [1, 2, 2, 3]),
                 ("thresholds", [1, 2, 3, float("nan")])]
        for key, original in (("temperatures", self.config["temperatures"]),
                              ("weights", self.config["weights"])):
            for value in (float("nan"), float("inf"), -1.0):
                cases.append((key, {**original, name: value}))
        cases += [("temperatures", {**self.config["temperatures"], name: 0}),
                  ("weights", {n: 0 for n in self.config["runs"]})]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(SystemExit):
                self.load({**self.config, key: value})
        for key in self.config:
            incomplete = {k: v for k, v in self.config.items() if k != key}
            with self.subTest(missing=key), self.assertRaises(SystemExit):
                self.load(incomplete)

    def test_cli_rejects_ensemble_evaluation_before_loading_models(self):
        cases = [["--ensemble", "ensemble.json"],
                 ["--evaluate"],
                 ["--evaluate", "--checkpoint", "a.pt", "b.pt"],
                 ["--evaluate", "--checkpoint", "a.pt", "--ensemble", "ensemble.json"],
                 ["--checkpoint", "a.pt", "a.pt"]]
        for args in cases:
            with self.subTest(args=args), patch.object(sys, "argv", ["demo.py", *args]), \
                    patch.object(demo, "Grader") as grader, contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                demo.main()
            self.assertEqual(error.exception.code, 2)
            grader.assert_not_called()
        with patch.object(sys, "argv", ["demo.py", "--evaluate", "--checkpoint", "a.pt", "--tta"]):
            self.assertTrue(demo.parse_args().tta)
        with patch.object(sys, "argv", ["demo.py"]):
            self.assertIsNone(demo.parse_args().tta)

    def test_missing_checkpoint_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(torch, "load") as load:
            with self.assertRaisesRegex(SystemExit, "không thấy checkpoint"):
                demo.Grader(Path(directory) / "missing.pt")
            load.assert_not_called()

    def test_ood_uses_base_view_while_logits_use_all_tta_views(self):
        grader = demo.Grader.__new__(demo.Grader)
        grader.device = torch.device("cpu")
        grader.model = SimpleNamespace(backbone=torch.nn.Flatten(1), head=torch.nn.Linear(4, 5))
        batch = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
        logits, features = grader.forward(batch, tta=True)
        reference = []
        with torch.no_grad():
            for k in range(8):
                view = torch.rot90(batch, k % 4, dims=(2, 3))
                if k >= 4:
                    view = torch.flip(view, dims=(3,))
                reference.append(grader.model.head(grader.model.backbone(view)))
        np.testing.assert_allclose(logits, torch.stack(reference).mean(0).numpy(), atol=1e-6)
        np.testing.assert_array_equal(features, batch.flatten(1).numpy())
        np.testing.assert_array_equal(features, grader.forward(batch, tta=False)[1])

    def test_saved_baseline_logits_reproduce_report_without_refitting(self):
        directory = ROOT / "results" / "baseline_no_mfiddr" / "checkpoints"
        config_path = directory / "ensemble_tta.json"
        if not config_path.is_file():
            self.skipTest("optional baseline artifacts not present")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        graders, all_logits, labels = [], [], None
        for name in config["runs"]:
            with np.load(directory / f"logits_{name}_test_tta.npz", allow_pickle=False) as archive:
                logits = archive["logits"]
                if labels is None:
                    labels = archive["labels"]
                else:
                    np.testing.assert_array_equal(labels, archive["labels"])
            graders.append(FakeGrader(name, logits[0]))
            all_logits.append(logits)
        ensemble = demo.Ensemble.from_json(graders, config_path)
        blended = sum(ensemble.weights[g.name] * g.probabilities(z / ensemble.temperatures[g.name])
                      for g, z in zip(graders, all_logits))
        predictions = {"test_argmax": blended.argmax(1),
                       "test_ordinal": np.array([ensemble.decode(p) for p in blended])}
        for variant, pred in predictions.items():
            for metric, value in demo.M.summarise(labels, pred).items():
                self.assertAlmostEqual(value, config["results"][variant][metric], places=12,
                                       msg=f"{variant}/{metric}")


if __name__ == "__main__":
    unittest.main()
