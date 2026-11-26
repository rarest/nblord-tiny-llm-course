"""The public ZIP must verify its training inputs before writing results."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class DownloadedTrainingTests(unittest.TestCase):
    def test_both_entrypoints_identify_changed_model_without_git(self):
        with TemporaryDirectory() as folder:
            package = Path(folder) / 'download'
            experiments = Path(__file__).resolve().parents[1]
            shutil.copytree(experiments, package / 'experiments',
                            ignore=shutil.ignore_patterns('__pycache__'))
            relative = 'experiments/transformer_depth/model.py'
            model = package / relative
            model.write_bytes(model.read_bytes() + b'\n')
            output = Path(folder) / 'my-training'
            for script in ('checkpoint_probe.py', 'limited_data_probe.py'):
                with self.subTest(script=script):
                    run = subprocess.run(
                        [sys.executable, '-B', str(package / 'experiments/checkpoint_selection' / script),
                         '--output', str(output)], cwd=folder,
                        env={**os.environ, 'PATH': '', 'PYTHONPATH': ''},
                        capture_output=True, text=True)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn(relative, run.stderr)
                    self.assertFalse(output.exists())
