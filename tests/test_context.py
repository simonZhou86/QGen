import unittest

from src.context import _chunk_text


class TestContext(unittest.TestCase):
    def test_chunk_text_respects_limit(self) -> None:
        text = "A" * 50 + "\n" + "B" * 50 + "\n" + "C" * 50
        chunks = _chunk_text(text, max_chars=60)
        self.assertTrue(len(chunks) >= 2)
        self.assertTrue(all(len(chunk) <= 60 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
