import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
from scipy.io.wavfile import read, write
from sokhanvar.engine import Engine, DEFAULT_VOICE


class FakeTTS:
    sample_rate = 24000

    def __init__(self):
        self.inputs = []
        self.frames = []
        self.voice = None
        self.flow_lm = SimpleNamespace(conditioner=SimpleNamespace(
            tokenizer=SimpleNamespace(sp=SimpleNamespace(
                encode=lambda text, out_type: list(range(1, len(text.split()) + 1)), unk_id=lambda: 0))))

    def get_state_for_audio_prompt(self, voice):
        self.voice = voice
        return {}

    def _generate_audio_stream_short_text(self, **kwargs):
        assert isinstance(kwargs["stop"], threading.Event)
        assert not kwargs["stop"].is_set()
        self.inputs.append(kwargs["text_to_generate"])
        self.frames.append(kwargs["frames_after_eos"])
        yield torch.ones(2400) * 0.1


class EngineTests(unittest.TestCase):
    def setUp(self):
        downloader = patch("pocket_tts.utils.utils.download_if_necessary", return_value="/missing-test-reference.wav")
        downloader.start()
        self.addCleanup(downloader.stop)

    def test_long_reference_is_capped_resampled_and_original_preserved(self):
        engine = Engine()
        engine.tts = FakeTTS()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "reference.wav"
            original = np.full(8 * 44100, 3000, dtype=np.int16)
            write(source, 44100, original)
            with patch.object(engine, "load_tts"), patch.dict("os.environ", {"PERSIAN_TTS_OUTPUT_DIR": folder}):
                _, _, report = engine.render([["سلام", "salAm", 0]],
                    str(source), 42, 18, 0.3, "A", frames_after_eos=4)
            self.assertEqual(tuple(engine.tts.voice.shape), (1, 5 * 24000))
            self.assertEqual(len(read(source)[1]), len(original))
            self.assertEqual(report["reference_duration_seconds"], 8)
            self.assertEqual(report["reference_used_seconds"], 5)
            self.assertEqual(engine.tts.frames, [4])

    def test_exact_phonemes_and_inserted_gap(self):
        engine = Engine()
        engine.tts = FakeTTS()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(engine, "load_tts"), patch.dict("os.environ", {"PERSIAN_TTS_OUTPUT_DIR": folder}):
                wav, metadata, report = engine.render(
                    [["اقتصاد آمریکا", "?eqtesAde1 ?AmrikA", 150],
                     ["سلام", "salAm", 900]], None, 42, 18, 0.3, "A")
            rate, audio = read(wav)
            self.assertEqual(rate, 24000)
            self.assertEqual(engine.tts.inputs, ["?eqtesAde ?AmrikA", "salAm"])
            self.assertEqual(engine.tts.eos_threshold, -2.0)
            self.assertEqual(engine.tts.frames, [3, 3])
            self.assertEqual(engine.tts.voice, DEFAULT_VOICE)
            self.assertEqual(len(audio), 2400 + 3600 + 2400)
            self.assertTrue((audio[2400:6000] == 0).all())
            self.assertEqual([r["inserted_pause_ms"] for r in report["chunks"]], [150, 0])
            self.assertEqual(json.loads(Path(metadata).read_text())["seed"], 42)

    def test_load_uses_persian_eos_setting(self):
        model = SimpleNamespace(capitalize_first_letter=False,
                                append_terminal_punctuation=False,
                                pad_with_spaces_for_short_inputs=False)
        with patch("pocket_tts.TTSModel.load_model", return_value=model) as load:
            Engine().load_tts()
        self.assertEqual(load.call_args.kwargs["eos_threshold"], -2.0)

    def test_custom_eos_and_upload_take_precedence(self):
        engine = Engine()
        engine.tts = FakeTTS()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(engine, "load_tts"), patch.dict("os.environ", {"PERSIAN_TTS_OUTPUT_DIR": folder}):
                _, _, report = engine.render([["سلام", "salAm", 0]],
                    "uploaded.wav", 42, 18, 0.3, "A", -1.5, "preset.wav")
        self.assertEqual(engine.tts.voice, "uploaded.wav")
        self.assertEqual(engine.tts.eos_threshold, -1.5)
        self.assertEqual(report["eos_threshold"], -1.5)

    def test_retries_silent_output_then_errors(self):
        engine = Engine()
        engine.tts = FakeTTS()
        engine.tts._generate_audio_stream_short_text = lambda **kwargs: iter([torch.zeros(2400)])
        with patch.object(engine, "load_tts"), self.assertRaisesRegex(RuntimeError, "attempts"):
            engine.render([["سلام", "salAm", 0]], None, 42, 18, 0.3, "A")

    def test_rejects_invalid_pause_before_model_load(self):
        engine = Engine()
        with self.assertRaisesRegex(ValueError, "pause"):
            engine.render([["سلام", "salAm", float("nan")]], None, 42, 18, 0.3, "A")
        self.assertIsNone(engine.tts)


if __name__ == "__main__":
    unittest.main()
