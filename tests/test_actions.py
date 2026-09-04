import unittest

from sap_rl_lab.actions import Action, ActionCodec, ActionKind


class ActionCodecTests(unittest.TestCase):
    def test_mapping_is_unique_and_round_trips(self):
        codec = ActionCodec(max_team_size=5, max_shop_size=5)
        self.assertEqual(codec.size, 71)
        for action_id in range(codec.size):
            self.assertEqual(codec.encode(codec.decode(action_id)), action_id)

    def test_adjacent_swaps_replace_factorial_permutations(self):
        codec = ActionCodec()
        swaps = [
            codec.decode(index)
            for index in range(codec.size)
            if codec.decode(index).kind is ActionKind.SWAP
        ]
        self.assertEqual(
            swaps,
            [
                Action(ActionKind.SWAP, 0, 1),
                Action(ActionKind.SWAP, 1, 2),
                Action(ActionKind.SWAP, 2, 3),
                Action(ActionKind.SWAP, 3, 4),
            ],
        )


if __name__ == "__main__":
    unittest.main()
