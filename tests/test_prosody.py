import unittest
from sokhanvar.prosody import chunk_phonemes, split_persian, validate_phonemes
from sokhanvar.normalize_fa import normalize


class ProsodyTests(unittest.TestCase):
    def test_boundary_changes_meaning(self):
        a = split_persian("بخشش، لازم نیست اعدامش کنید.")
        b = split_persian("بخشش لازم نیست، اعدامش کنید.")
        self.assertEqual([s.text for s in a], ["بخشش", "لازم نیست اعدامش کنید"])
        self.assertEqual([s.text for s in b], ["بخشش لازم نیست", "اعدامش کنید"])
        self.assertEqual([s.pause_ms for s in a], [150, 0])

    def test_no_comma_split_and_sentence_without_space(self):
        rows = split_persian("سلام، دوست من.خوبی؟", use_commas=False)
        self.assertEqual([r.text for r in rows], ["سلام دوست من", "خوبی"])
        self.assertEqual([r.pause_ms for r in rows], [250, 0])

    def test_ezafe_chain_never_split(self):
        count = lambda s: len(s.split())
        self.assertEqual(chunk_phonemes("in ketAbe1 duste1 man ast", count, 3),
                         ["in", "ketAbe1 duste1 man", "ast"])
        with self.assertRaises(ValueError):
            chunk_phonemes("ketAbe1 duste1 man", count, 2)
        with self.assertRaises(ValueError):
            chunk_phonemes("ketAbe1", count)

    def test_question_phoneme_and_zh_survive(self):
        validate_phonemes("?eqtesAde1 ?AmrikA ;Ale")
        for invalid in ("سلام", "salAm,", "salAm.", "", "1salAm", "salAm11", "1"):
            with self.assertRaises(ValueError):
                validate_phonemes(invalid)

    def test_normalization_keeps_zwnj_and_spells_digits(self):
        self.assertEqual(normalize("كتاب من تا سال ۲۰۳۰ مي‌رود."),
                         "کتاب من تا سال دو هزار و سی می‌رود.")


if __name__ == "__main__":
    unittest.main()
