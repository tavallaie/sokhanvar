import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sokhanvar.reference import (UPLOAD_CHOICE, persist_reference,
                                   saved_reference, clear_reference, resolve_reference)


class ReferenceTests(unittest.TestCase):
    def test_upload_survives_reload_and_keeps_exact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'my-voice.wav'
            source.write_bytes(b'user reference audio')
            with patch('sokhanvar.reference.REFERENCE_DIR', Path(folder) / 'stored'):
                selected = persist_reference(source)
                source.unlink()
                self.assertEqual(saved_reference(), selected)
                self.assertEqual(Path(selected).read_bytes(), b'user reference audio')
                clear_reference()
                self.assertIsNone(saved_reference())

    def test_missing_upload_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, 'reference audio is missing'):
            resolve_reference(None, UPLOAD_CHOICE, {'sample': 'sample.wav'})

    def test_sample_requires_explicit_selection(self):
        self.assertEqual(resolve_reference('uploaded.wav', 'sample', {'sample': 'sample.wav'}),
                         (None, 'sample.wav'))

    def test_selected_upload_wins_over_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'my-voice.wav'
            source.write_bytes(b'user reference audio')
            with patch('sokhanvar.reference.REFERENCE_DIR', Path(folder) / 'stored'):
                selected, _ = resolve_reference(source, UPLOAD_CHOICE, {'sample': 'sample.wav'})
                self.assertEqual(Path(selected).read_bytes(), source.read_bytes())


if __name__ == '__main__':
    unittest.main()
