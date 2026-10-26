"""Test-runner entry point for 04_fst_noisy_simulator.py.

The simulator's self-tests live inside the script (class EngineSelfTests) so that
`python 04_fst_noisy_simulator.py --selftest` works on its own. Because the module name
starts with a digit it cannot be imported normally, so this file loads it by path
and re-exports the suite for standard discovery:

    python -m pytest
    python -m unittest discover -s tests -v
"""

import importlib.util
import pathlib
import sys
import unittest

ENGINE_PATH = pathlib.Path(__file__).resolve().parent.parent / "04_fst_noisy_simulator.py"

_spec = importlib.util.spec_from_file_location("fst_noisy_simulator", ENGINE_PATH)
engine = importlib.util.module_from_spec(_spec)
sys.modules["fst_noisy_simulator"] = engine  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(engine)


class TestFSTNoisySimulator(engine.EngineSelfTests):
    """All self-tests (FST transitions, slot memory, channel constraints, entropy, hazards, probes)."""


class TestEndToEndRun(unittest.TestCase):
    """Runs the full simulator offline, exactly as `python 04_fst_noisy_simulator.py` does."""

    def test_full_offline_run_exits_cleanly(self):
        import contextlib
        import io
        import os
        import tempfile

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            md = os.path.join(tmp, "transcript.md")
            with contextlib.redirect_stdout(out):
                code = engine.main(["--llm", "mock", "--no-color", "--quiet-rpc", "--transcript", md])
            with open(md, encoding="utf-8") as fh:
                transcript = fh.read()
        log = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("MISSION 2 EXECUTION COMPLETE", log)
        for heading in ("## 1. FST test", "## 2. Anti-Turing probe", "## 3. 10-turn Noisy Interrogation",
                        "## 4. Channel entropy metrics", "## 6. Full console log"):
            self.assertIn(heading, transcript)
        self.assertEqual(transcript.count("\n| 10 |"), 1, "the interrogation table has turns 1-10")


if __name__ == "__main__":
    unittest.main()
