"""CPU checks for the fixed-CFG PuLID experiment; no model loading."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import plot_canonical
import run_pulid_matched as experiment
import run_sweep


class PuLIDSweepTests(unittest.TestCase):
    def test_actual_inference_commands_have_fixed_cfg_and_six_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'not_created'
            args = experiment.parse_args(['--output-root', str(output)])
            commands = experiment.build_commands(args)
            command = commands[1][1]
            log = io.StringIO()
            with patch.object(sys, 'argv', [command[2], *command[3:], '--dry-run']), \
                    patch.object(run_sweep.subprocess, 'run') as launch, redirect_stdout(log):
                run_sweep.main()
            launch.assert_not_called()
            rendered = [shlex.split(line) for line in log.getvalue().splitlines() if ' -m torch.distributed.run ' in line]
            self.assertEqual(len(rendered), 6)
            self.assertEqual([float(c[c.index('--id_weight')+1]) for c in rendered], list(experiment.WEIGHTS))
            for c in rendered:
                self.assertEqual(float(c[c.index('--id_guidance_scale')+1]), 1.2)
                self.assertEqual(c[c.index('--checkpoint_path')+1], str(experiment.ROOT/'checkpoints/student'))
                self.assertEqual(c[c.index('--max_pairs')+1], '0')
                self.assertNotIn('--condition_type', c)
            names = [f'pulid_cfg1.2_w{w:g}_steps28_seed42' for w in experiment.WEIGHTS]
            for _, scoring in commands[2:4]:
                start = scoring.index('--run-names') + 1
                end = scoring.index('--real-features') if '--real-features' in scoring else len(scoring)
                self.assertEqual(scoring[start:end], names)
            self.assertFalse(output.exists())

    def test_dry_run_never_launches_any_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'not_created'
            with patch.object(experiment.subprocess, 'run') as launch, redirect_stdout(io.StringIO()):
                experiment.main(['--dry-run', '--output-root', str(output)])
            launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_original_cfg_results_are_protected(self):
        with redirect_stdout(io.StringIO()), patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                experiment.parse_args(['--output-root', str(experiment.HERE/'outputs/student_matched_full1000')])

    def test_plot_exports_weight_column_and_combined_figure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, fid = [], []
            for weight in experiment.WEIGHTS:
                for pair in ['000000', '000001']:
                    rows.append(dict(stage='pulid', id_guidance_scale=1.2, id_weight=weight,
                                     pair_id=pair, success=True, id_similarity=weight/2,
                                     top1=1, top5=1, pose=weight+1, exp=weight))
                fid.append(dict(stage='pulid', id_guidance_scale=1.2, id_weight=weight,
                                fid=weight+2, fid_n_generated=2, fid_n_real=70000))
            pd.DataFrame(rows).to_csv(root/'scores.csv', index=False)
            pd.DataFrame(fid).to_csv(root/'fid.csv', index=False)
            with patch.object(sys, 'argv', ['plot_canonical.py', '--scores', str(root/'scores.csv'),
                                           '--out-dir', str(root/'figures'), '--combined']), redirect_stdout(io.StringIO()):
                plot_canonical.main()
            result = pd.read_csv(root/'figures/metrics_table.csv')
            self.assertEqual(result.id_weight.tolist(), list(experiment.WEIGHTS))
            self.assertTrue(result.w.eq(1.2).all())
            self.assertTrue(result.n_common.eq(2).all())
            self.assertTrue((root/'figures/combined_tradeoffs.png').stat().st_size > 1000)
            self.assertTrue((root/'figures/combined_tradeoffs.svg').is_file())


if __name__ == '__main__':
    unittest.main()
