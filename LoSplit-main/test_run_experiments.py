"""Launcher report checks without importing torch or running training."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiments as runner


class ReportTests(unittest.TestCase):
    def test_final_markers_and_attack_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "task.log"
            log.write_text("****After GTA Attack****\nASR: 0.900000/100 nodes\nClean Accuracy: 0.800000\n", encoding="utf-8")
            self.assertEqual(runner.parse_metrics(log, "attack"), {"ASR": .9, "CA": .8})
            self.assertEqual(runner.parse_metrics(log, "prune"), {})
            log.write_text("ASR: 0.99\nFINAL_DEFENSE_RESULT method=prune CA=0.8 ASR=0.1\n"
                           "FINAL_RIGBD_RESULT K=20 b=0.5 CA=0.7 ASR=0.2 Precision=1 Recall=0.5 FPR=0\n",
                           encoding="utf-8")
            self.assertEqual(runner.parse_metrics(log, "prune"), {"CA": .8, "ASR": .1})
            self.assertEqual(runner.parse_metrics(log, "rigbd"),
                             {"CA": .7, "ASR": .2, "Precision": 1., "Recall": .5, "FPR": 0.})

    def simulate(self, root, failure=None):
        arguments = ["run_experiments.py", "--attacks", "GTA", "--methods", "none", "prune",
                     "--attack-root", str(root / "attacks"), "--defense-root", str(root / "defenses")]

        def fake_job(job, env):
            self.assertEqual(env["PYTHONHASHSEED"], "10")
            name, entry, flags, outputs, inputs, log = job
            for path in inputs:
                self.assertTrue(path.is_file())
                self.assertIn((root / "attacks").resolve(), path.parents)
            if name.endswith("/none") and failure:
                raise failure
            for path in outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            content = ("****After GTA Attack****\nASR: 0.9\nClean Accuracy: 0.8\n"
                       if name.endswith("/attack") else "FINAL_DEFENSE_RESULT method=none CA=0.8 ASR=0.1\n")
            log.write_text(content, encoding="utf-8")

        with patch("sys.argv", arguments), patch.object(runner.subprocess, "run"), \
                patch.object(runner, "run_job", side_effect=fake_job), contextlib.redirect_stdout(io.StringIO()):
            if failure:
                with self.assertRaises(type(failure)):
                    runner.main()
            else:
                runner.main()
        return (root / "defenses" / "results.md").read_text(encoding="utf-8")

    def test_all_pairing_and_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.simulate(Path(directory))
            self.assertIn("全部完成", report)
            self.assertIn("90.0000", report)
            self.assertIn("10.0000", report)
            self.assertIn("GTA/attack", report)
            self.assertIn("GTA/prune", report)

    def test_failure_and_interrupt_preserve_results(self):
        for error in (RuntimeError("training failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                report = self.simulate(Path(directory), error)
                self.assertIn("90.0000", report)
                self.assertIn("未运行", report)
                self.assertIn("中断" if isinstance(error, KeyboardInterrupt) else "training failed", report)

    def test_existing_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "existing.md"
            report.write_text("keep me", encoding="utf-8")
            with patch("sys.argv", ["runner", "--report", str(report)]), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.main()
            self.assertEqual(report.read_text(encoding="utf-8"), "keep me")


if __name__ == "__main__":
    unittest.main()
