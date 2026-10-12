"""Test-runner entry point for 04_dialogic_engine.py.

The engine's self-tests live inside the engine (class EngineSelfTests) so that
`python 04_dialogic_engine.py --selftest` works on its own. Because the module name
starts with a digit it cannot be imported normally, so this file loads it by path
and re-exports the suite for standard discovery:

    python -m pytest
    python -m unittest discover -s tests -v
"""

import importlib.util
import pathlib
import sys
import unittest

ENGINE_PATH = pathlib.Path(__file__).resolve().parent.parent / "04_dialogic_engine.py"

_spec = importlib.util.spec_from_file_location("dialogic_engine", ENGINE_PATH)
engine = importlib.util.module_from_spec(_spec)
sys.modules["dialogic_engine"] = engine  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(engine)


class TestDialogicEngine(engine.EngineSelfTests):
    """All engine self-tests (state transitions, aporia, debate invariants, fallbacks)."""


class TestEndToEndRun(unittest.TestCase):
    """Runs the full engine offline, exactly as `python 04_dialogic_engine.py` does."""

    def test_full_offline_run_exits_cleanly(self):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = engine.main(["--llm", "mock", "--no-color", "--quiet-rpc"])
        log = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("MISSION 1 EXECUTION COMPLETE", log)
        self.assertIn("APORIA", log.upper())
        self.assertIn("CONDITIONAL AUTHORIZATION", log.upper())


if __name__ == "__main__":
    unittest.main()
