import tempfile
import unittest
from pathlib import Path

from sap_rl_lab.actions import ActionKind
from sap_rl_lab.engine import AutoBattler
from sap_rl_lab.replay import EpisodeReplay, ReplayRecorder, verify_replay


class ReplayTests(unittest.TestCase):
    def test_round_trip_and_deterministic_verification(self):
        engine = AutoBattler()
        recorder = ReplayRecorder(engine, seed=83)
        buy_id = next(
            engine.codec.encode(action)
            for action in engine.legal_actions()
            if action.kind is ActionKind.BUY_PET
        )
        recorder.step(buy_id)
        end_id = next(
            engine.codec.encode(action)
            for action in engine.legal_actions()
            if action.kind is ActionKind.END_TURN
        )
        recorder.step(end_id)
        replay = recorder.finish()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.json"
            replay.save(path)
            loaded = EpisodeReplay.load(path)

        verify_replay(loaded)
        self.assertEqual(loaded.to_dict(), replay.to_dict())
        self.assertTrue(any(step.info.get("battle_trace") for step in loaded.steps))

    def test_changed_step_is_detected(self):
        recorder = ReplayRecorder(AutoBattler(), seed=5)
        end_id = next(
            recorder.engine.codec.encode(action)
            for action in recorder.engine.legal_actions()
            if action.kind is ActionKind.END_TURN
        )
        recorder.step(end_id)
        replay = recorder.finish()
        replay.final_state["gold"] = 999
        with self.assertRaisesRegex(AssertionError, "differs"):
            verify_replay(replay)


if __name__ == "__main__":
    unittest.main()
