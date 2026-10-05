"""Exercise a native Persian backend without any ML runtime."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import wave

from sokhanvar import (BackendInfo, Phrase, Sokhanvar, SokhanvarConfig,
                       SpeechPlan, SynthesisResult, available_backends, register_backend)
from sokhanvar.backends import registry


class NativePersianBackend:
    info = BackendInfo('native_fa_test', input_kind='text')

    def __init__(self, config):
        self.model = config.model

    def validate_config(self, config):
        if set(config.backend_options) - {'speed'}:
            raise ValueError('Unknown native backend option')

    def prepare(self, text, config):
        return SpeechPlan(text, [Phrase(text)])

    def synthesize(self, plan, config):
        folder = Path(config.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'native.wav'
        with wave.open(str(path), 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b'\x00\x00' * 240)
        return SynthesisResult(path, path.with_suffix('.json'),
                               {'text': plan.phrases[0].text, 'options': config.backend_options})


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.factories = patch.dict(registry._factories)
        self.factories.start()
        register_backend('native_fa_test', NativePersianBackend)

    def tearDown(self):
        self.factories.stop()

    def config(self, **kwargs):
        return SokhanvarConfig(backend='native_fa_test', model='test/persian-model', **kwargs)

    def test_native_persian_round_trip_without_reference_or_phonemes(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(output_dir=folder, token_budget=128, backend_options={'speed': 1.2})
            config.to_yaml(Path(folder) / 'config.yaml')
            speaker = Sokhanvar.from_yaml(Path(folder) / 'config.yaml')
            plan = speaker.prepare('کتاب دوست من روی میز است.')
            self.assertEqual(plan.phrases[0].phonemes, '')
            self.assertEqual(plan.backend, config.backend)
            plan.to_yaml(Path(folder) / 'plan.yaml')
            result = speaker.synthesize_plan(SpeechPlan.from_yaml(Path(folder) / 'plan.yaml'), Path(folder) / 'chosen.wav')
            self.assertEqual(result.metadata['text'], plan.normalized_text)
            self.assertEqual(result.metadata['options'], {'speed': 1.2})
            self.assertEqual(result.metadata['model'], config.model)
            self.assertEqual(result.metadata['language'], 'fa')
            self.assertEqual(json.loads(result.metadata_path.read_text()), result.metadata)
            with wave.open(str(result.audio_path)) as audio:
                self.assertEqual(audio.getnframes(), 240)

    def test_unknown_backend_fails_without_fallback(self):
        with self.assertRaisesRegex(ValueError, 'Unknown backend'):
            Sokhanvar(SokhanvarConfig(backend='missing_backend', model='some/persian-model'))

    def test_backend_options_validated_by_adapter(self):
        with self.assertRaisesRegex(ValueError, 'Unknown native backend option'):
            Sokhanvar(self.config(backend_options={'typo': 1}))

    def test_persian_language_requirement(self):
        with self.assertRaisesRegex(ValueError, 'language must be fa'):
            self.config(language='en')

    def test_additional_backend_requires_explicit_model(self):
        with self.assertRaisesRegex(ValueError, 'model must be a nonempty string'):
            SokhanvarConfig(backend='native_fa_test')

    def test_selected_model_controls_pocket_weights_uri(self):
        config = SokhanvarConfig(model='owner/another-persian-pocket')
        self.assertEqual(config.model_config, 'hf://owner/another-persian-pocket/model.yaml')
        with self.assertRaisesRegex(ValueError, 'selected model'):
            SokhanvarConfig(model='owner/new-model', model_config='hf://owner/old-model/model.yaml')

    def test_plan_cannot_cross_backend_or_model(self):
        speaker = Sokhanvar(self.config())
        for plan in (SpeechPlan('', [], backend='pocket_tts_farsi'),
                     SpeechPlan('', [], model='other/model')):
            with self.assertRaisesRegex(ValueError, 'another'):
                speaker.synthesize_plan(plan)

    def test_entry_point_discovery_is_lazy(self):
        plugin = Mock(name='plugin')
        plugin.name = 'native_fa_test'
        plugin.load.return_value = NativePersianBackend
        with patch.dict(registry._factories, {}, clear=True), patch.object(registry, '_plugins', return_value=[plugin]):
            self.assertEqual(available_backends(), ['native_fa_test'])
            plugin.load.assert_not_called()
            Sokhanvar(self.config())
            plugin.load.assert_called_once()

    def test_duplicate_registration_rejected(self):
        with self.assertRaisesRegex(ValueError, 'already registered'):
            register_backend('native_fa_test', NativePersianBackend)

    def test_playground_routes_native_text_generation_and_export(self):
        from sokhanvar.app import build_app
        with tempfile.TemporaryDirectory() as folder:
            app = build_app(self.config(output_dir=folder, token_budget=128, backend_options={'speed': 1.1}))
            callbacks = {getattr(fn.fn, '__name__', ''): fn.fn for fn in app.fns.values()}
            norm, a, b, *_ = callbacks['convert']('سلام دنیا', 150, 250, False)
            self.assertEqual(a[0], ['سلام دنیا', '', 0])
            with patch('sokhanvar.app.resolve_reference', side_effect=AssertionError('Unexpected voice lookup')):
                wav, _, report = callbacks['generate_both'](a, b, None, 42, 18, .3, -2, 'My uploaded reference', 3)[:3]
            self.assertTrue(Path(wav).is_file())
            self.assertEqual(report['backend'], 'native_fa_test')
            config_file, plan_file, _ = callbacks['export_settings'](a, b, 'A', norm, None, 'My uploaded reference', 42, 18, .3, -2, 3, 150, 250, False)
            self.assertEqual(SokhanvarConfig.from_yaml(config_file).backend_options, {'speed': 1.1})
            self.assertEqual(SokhanvarConfig.from_yaml(config_file).token_budget, 128)
            self.assertEqual(SpeechPlan.from_yaml(plan_file).backend, 'native_fa_test')


if __name__ == '__main__':
    unittest.main()
