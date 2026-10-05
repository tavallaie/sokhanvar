import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import yaml
from sokhanvar import Sokhanvar, SokhanvarConfig, SpeechPlan, Phrase
from sokhanvar.export import export_bundle


class LibraryTests(unittest.TestCase):
    def test_import_does_not_load_gradio_or_models(self):
        result = subprocess.run([sys.executable, '-c',
            'import sys; import sokhanvar; assert "gradio" not in sys.modules; assert "torch" not in sys.modules'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_yaml_validates_unknown_and_unsafe_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            for content in ('typo: 1', 'frames_after_eos: -1', 'split_at_commas: "false"',
                            'temperature: .nan', '- not-a-mapping'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    SokhanvarConfig.from_yaml(path)

    def test_yaml_paths_resolve_beside_config_not_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            path.write_text('reference_audio: voices/ref.wav\noutput_dir: audio\n')
            config = SokhanvarConfig.from_yaml(path)
            self.assertEqual(config.reference_audio, str(Path(folder) / 'voices/ref.wav'))
            self.assertEqual(config.output_dir, str(Path(folder) / 'audio'))
            self.assertEqual(config.frames_after_eos, 3)

    def test_missing_reference_fails_before_loading_g2p(self):
        speaker = Sokhanvar()
        with patch.object(speaker._engine, 'prepare') as prepare:
            with self.assertRaisesRegex(ValueError, 'reference_audio'):
                speaker.synthesize('سلام')
            prepare.assert_not_called()

    def test_bundle_reloads_after_move_and_preserves_edited_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / 'my.wav'
            reference.write_bytes(b'exact original reference')
            config = SokhanvarConfig(reference_audio=str(reference), frames_after_eos=4, eos_threshold=-1.5)
            plan = SpeechPlan('کتاب من', [Phrase('کتاب من', 'ketAbe1 man', 175)])
            _, _, bundle = export_bundle(config, plan)
            destination = Path(folder) / 'other-project'
            with zipfile.ZipFile(bundle) as archive:
                archive.extractall(destination)
            restored = SokhanvarConfig.from_yaml(destination / 'sokhanvar.yaml')
            self.assertEqual(Path(restored.reference_audio).read_bytes(), reference.read_bytes())
            self.assertEqual(restored.frames_after_eos, 4)
            self.assertEqual(restored.eos_threshold, -1.5)
            self.assertEqual(SpeechPlan.from_yaml(destination / 'plan.yaml'), plan)

    def test_library_passes_yaml_settings_and_returns_requested_files(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            reference = folder / 'ref.wav'
            reference.write_bytes(b'reference')
            config = SokhanvarConfig(reference_audio=str(reference), seed=9,
                token_budget=12, temperature=0.4, eos_threshold=-1, frames_after_eos=4)
            speaker = Sokhanvar(config)
            wav, meta = folder / 'generated.wav', folder / 'generated.json'
            wav.write_bytes(b'wave'); meta.write_text('{}')
            plan = SpeechPlan('سلام', [Phrase('سلام', 'salAm')])
            with patch.object(speaker._engine, 'render', return_value=(str(wav), str(meta), {})) as render:
                result = speaker.synthesize_plan(plan, folder / 'chosen.wav')
            self.assertEqual(result.audio_path.read_bytes(), b'wave')
            self.assertEqual(result.metadata_path, folder / 'chosen.json')
            self.assertEqual(render.call_args.args[2:5], (9, 12, 0.4))
            self.assertEqual(render.call_args.args[-3:], (-1, str(reference), 4))

    def test_yaml_save_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            config = SokhanvarConfig(reference_audio='hf://example/voice.wav', output_dir=str(Path(folder) / 'out'))
            path = config.to_yaml(Path(folder) / 'settings.yaml')
            self.assertEqual(SokhanvarConfig.from_yaml(path), config)


if __name__ == '__main__':
    unittest.main()
