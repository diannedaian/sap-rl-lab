"""Read-only battle observation must not alter the simulated trajectory."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import sap_rl_lab.engine as engine_module
from sap_rl_lab.domain import Pet
from sap_rl_lab.engine import AutoBattler

spec = importlib.util.spec_from_file_location(
    "export_replay_viewer", Path(__file__).parents[1] / "scripts/export_replay_viewer.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ReplayViewerTests(unittest.TestCase):
    def test_confirmation_selection_preserves_failures_and_pairs_initial_seeds(self):
        names = ["unshaped-seed503", "swap_cost-seed503"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "release").mkdir()
            (root / "release" / "manifest.json").write_text(
                json.dumps({"model": names[1], "seed": 503})
            )
            for name in names:
                rows = [
                    {
                        "seed": 1500000,
                        "truncated": name == names[0],
                        "success": name == names[1],
                        "final_turn": 8,
                    },
                    {"seed": 1500001, "truncated": True, "success": False, "final_turn": 9},
                    {"seed": 1500002, "truncated": False, "success": True, "final_turn": 10},
                ]
                (root / f"test-{name}.json").write_text(
                    json.dumps(
                        {
                            "families": {
                                f: {"episode_results": rows} for f in ("stats", "greedy", "summon")
                            }
                        }
                    )
                )
            cases, defaults = exporter.select_confirmation_cases(root, dict.fromkeys(names))
            assert defaults == [f"{name}-stats-1500000" for name in names]
            # The default pair is already present; it must not be duplicated.
            assert len(cases) == 6
            assert any(n == names[1] and c == "cutoff" for n, _, c, _ in cases)
            assert len({(n, f, r["seed"]) for n, f, _, r in cases}) == len(cases)

    def test_battle_observer_is_read_only_and_captures_frames(self):
        engines = [AutoBattler() for _ in range(2)]
        for engine in engines:
            engine.reset(4)
            engine.state.team = [Pet("cricket", 1, 3), Pet("horse", 2, 1)]
            engine.opponent_provider = lambda *args: [Pet("pig", 4, 1)]
        observer = exporter.BattleObserver(engine_module.__file__)
        previous_trace = sys.gettrace()
        try:
            sys.settrace(observer)
            observed = engines[0].step_id(0)
        finally:
            sys.settrace(previous_trace)
        expected = engines[1].step_id(0)
        self.assertEqual(observed, expected)
        self.assertEqual(engines[0].state.to_dict(), engines[1].state.to_dict())
        self.assertEqual(engines[0].rng.getstate(), engines[1].rng.getstate())
        self.assertGreater(len(observer.frames), 3)
        self.assertEqual(observer.frames[0]["event"], "Battle begins")
        self.assertEqual(observer.frames[0]["teams"][0][0]["health"], 3)
        self.assertTrue(any("summon" in f["event"] for f in observer.frames))

    def test_observer_accepts_symlinked_source_filenames(self):
        # macOS /var and /private/var refer to the same temporary directory.
        with tempfile.TemporaryDirectory() as temporary:
            alias = Path(temporary) / "engine-alias.py"
            alias.symlink_to(engine_module.__file__)
            observer = exporter.BattleObserver(engine_module.__file__)
            code = compile(
                "teams = [[], []]\ntrace = []\ntrace.append('done')\n", str(alias), "exec"
            )
            previous_trace = sys.gettrace()
            try:
                sys.settrace(observer)
                exec(code, {})
            finally:
                sys.settrace(previous_trace)
            self.assertEqual(observer.frames[-1]["event"], "done")


if __name__ == "__main__":
    unittest.main()
