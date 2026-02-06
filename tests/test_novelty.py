import unittest

from src.novelty import lexical_similarity


class TestNovelty(unittest.TestCase):
    def test_similarity_detects_overlap(self) -> None:
        a = "What was the primary endpoint in week 12?"
        b = "At week 12, what primary endpoint change was observed?"
        self.assertGreater(lexical_similarity(a, b), 0.3)

    def test_similarity_low_for_unrelated(self) -> None:
        a = "What is the terminal half-life of ZN-314?"
        b = "Explain randomization ratio in the trial design."
        self.assertLess(lexical_similarity(a, b), 0.5)


if __name__ == "__main__":
    unittest.main()
