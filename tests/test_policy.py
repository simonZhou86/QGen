import unittest

from src.policy import ema_update


class TestPolicy(unittest.TestCase):
    def test_ema_first_value(self) -> None:
        self.assertEqual(ema_update(None, 0.75, 0.2), 0.75)

    def test_ema_updates(self) -> None:
        updated = ema_update(0.5, 1.0, 0.2)
        self.assertAlmostEqual(updated, 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
