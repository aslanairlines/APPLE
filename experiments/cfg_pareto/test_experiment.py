"""CPU regression checks for sweep setup and fair paired comparisons."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

import evaluate
import patch_apple
import plot_pareto
import run_sweep


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        for folder in ("src", "trg"):
            (self.data / folder).mkdir(parents=True)
            for pair in ("000001", "000002"):
                (self.data / folder / f"{pair}.jpg").write_bytes(b"fixture")
        self.checkpoint = self.root / "student.safetensors"
        self.checkpoint.write_bytes(b"fixture weights")
        self.script = self.root / "inference.py"
        self.script.write_text("# CFG_PARETO_PATCH_V2\n")
        self.output = self.root / "outputs"
        self.argv = ["run_sweep.py", "--ffhq-base", str(self.data), "--checkpoint",
                     str(self.checkpoint), "--output-root", str(self.output),
                     "--cfg-scales", "1", "2", "--max-pairs", "2"]

    def run_sweep(self, extra=(), produced=("000001", "000002")):
        def fake_inference(command, **kwargs):
            self.assertEqual(command[:3], [sys.executable, "-m", "torch.distributed.run"])
            self.assertNotIn("--condition_type", command)
            dest = Path(command[command.index("--output_dir") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            for pair in produced:
                (dest / f"{pair}.png").write_bytes(b"fixture")
            return SimpleNamespace(returncode=0)

        with patch.object(sys, "argv", self.argv + list(extra)), \
                patch.object(run_sweep, "SCRIPT", self.script), \
                patch.object(run_sweep, "ROOT", self.root), \
                patch.object(run_sweep.subprocess, "run", side_effect=fake_inference), \
                redirect_stdout(io.StringIO()):
            run_sweep.main()

    def test_student_default_uses_raw_target_and_current_python(self):
        self.run_sweep()
        runs = evaluate.collect_runs(self.output, self.data)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0][1]["condition_type"], "raw")
        self.assertEqual(runs[0][1]["pair_ids"], ["000001", "000002"])

    def test_lfs_pointer_is_rejected(self):
        self.checkpoint.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(SystemExit, "Git LFS pointer"):
            self.run_sweep()

    def test_incomplete_outputs_never_marked_complete(self):
        with self.assertRaisesRegex(SystemExit, "Incomplete/contaminated"):
            self.run_sweep(produced=("000001",))
        manifest = next(self.output.glob("*/manifest.json"))
        self.assertEqual(json.loads(manifest.read_text())["status"], "failed_incomplete")

    def test_nan_guidance_rejected_before_launch(self):
        with self.assertRaisesRegex(SystemExit, "finite"):
            self.run_sweep(["--cfg-scales", "nan"])
        self.assertFalse(self.output.exists())

    def test_missing_target_rejected_before_launch(self):
        (self.data / "trg/000002.jpg").unlink()
        with self.assertRaisesRegex(SystemExit, "missing target"):
            self.run_sweep()

    def test_mixed_seeds_not_evaluated_together(self):
        self.run_sweep()
        manifest = list(self.output.glob("*/manifest.json"))[1]
        meta = json.loads(manifest.read_text())
        meta["seed"] += 1
        manifest.write_text(json.dumps(meta))
        with self.assertRaisesRegex(SystemExit, "not comparable.*seed"):
            evaluate.collect_runs(self.output, self.data)

    def test_completed_run_missing_output_rejected(self):
        self.run_sweep()
        next(self.output.glob("*/images/000002.png")).unlink()
        with self.assertRaisesRegex(SystemExit, "Missing or unexpected"):
            evaluate.collect_runs(self.output, self.data)

    def test_requested_unfinished_run_is_error(self):
        self.run_sweep()
        manifest = next(self.output.glob("*/manifest.json"))
        meta = json.loads(manifest.read_text())
        meta["status"] = "running"
        manifest.write_text(json.dumps(meta))
        with self.assertRaisesRegex(SystemExit, "unfinished"):
            evaluate.collect_runs(self.output, self.data, [manifest.parent.name])

    def test_landmarks_share_inference_resolution(self):
        image = self.root / "large.png"
        Image.new("RGB", (1024, 1024)).save(image)

        class Detector:
            def get_landmarks_from_image(self, rgb):
                self.shape = rgb.shape
                return [np.zeros((68, 2), dtype=np.float32)]

        detector = Detector()
        evaluate.landmarks(detector, image)
        self.assertEqual(detector.shape, (512, 512, 3))

    def test_patch_applies_and_is_idempotent(self):
        original = patch_apple.SCRIPT.read_text()
        if patch_apple.MARKER in original or patch_apple.LEGACY_MARKER in original:
            original = (patch_apple.ROOT / ".git/apple_inference_before_cfg_pareto.py").read_text()
        self.script.write_text(original)
        with patch.object(patch_apple, "SCRIPT", self.script), \
                patch.object(patch_apple, "ROOT", self.root), redirect_stdout(io.StringIO()):
            patch_apple.main()
            first = self.script.read_text()
            ast.parse(first)
            self.assertIn("torch.manual_seed(pair_seed)", first)
            self.assertIn("cfg_numpy.random.seed(pair_seed)", first)
            patch_apple.main()
            self.assertEqual(first, self.script.read_text())

    def test_legacy_patch_is_upgraded(self):
        original = patch_apple.SCRIPT.read_text()
        if patch_apple.MARKER in original or patch_apple.LEGACY_MARKER in original:
            original = (patch_apple.ROOT / ".git/apple_inference_before_cfg_pareto.py").read_text()
        legacy = patch_apple.patch_controls(original).replace(patch_apple.MARKER, patch_apple.LEGACY_MARKER)
        self.script.write_text(legacy)
        with patch.object(patch_apple, "SCRIPT", self.script), \
                patch.object(patch_apple, "ROOT", self.root), redirect_stdout(io.StringIO()):
            patch_apple.main()
        upgraded = self.script.read_text()
        ast.parse(upgraded)
        self.assertIn(patch_apple.MARKER, upgraded)
        self.assertEqual(upgraded.count('parser.add_argument("--id_weight"'), 1)
        self.assertIn("torch.manual_seed(pair_seed)", upgraded)

    def test_plot_compares_common_successful_pairs(self):
        data = pd.DataFrame([
            dict(stage="cfg", id_guidance_scale=1, id_weight=1, pair_id="a", success=True,
                 id_similarity=.4, landmark_error=.2),
            dict(stage="cfg", id_guidance_scale=1, id_weight=1, pair_id="b", success=True,
                 id_similarity=.9, landmark_error=.1),
            dict(stage="cfg", id_guidance_scale=2, id_weight=1, pair_id="a", success=True,
                 id_similarity=.5, landmark_error=.3),
            dict(stage="cfg", id_guidance_scale=2, id_weight=1, pair_id="b", success=False,
                 id_similarity=np.nan, landmark_error=np.nan),
        ])
        with redirect_stdout(io.StringIO()):
            summary = plot_pareto.summarize(data).set_index("id_guidance_scale")
        self.assertAlmostEqual(summary.loc[1, "id_similarity"], .4)
        self.assertEqual(summary.loc[1, "n_valid"], 1)
        self.assertEqual(summary.loc[1, "detection_success_rate"], 1)
        self.assertEqual(summary.loc[2, "detection_success_rate"], .5)
        data.loc[data.id_guidance_scale == 2, "success"] = False
        with self.assertRaisesRegex(SystemExit, "No common successful pairs"):
            plot_pareto.summarize(data)


if __name__ == "__main__":
    unittest.main()
