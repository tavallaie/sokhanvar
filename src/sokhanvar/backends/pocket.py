"""Adapter for the Persian phoneme-trained Pocket TTS models."""
from pathlib import Path
from .base import BackendInfo


class PocketFarsiBackend:
    def __init__(self, config):
        self.validate_config(config)
        try:
            from ..engine import Engine, VOICE_PRESETS, MODEL
        except ModuleNotFoundError as error:
            raise ImportError('Install the Pocket runtime: uv sync --extra pocket or pip install "sokhanvar[pocket]"') from error
        presets = {name: uri.replace(f"hf://{MODEL}/", f"hf://{config.model}/") for name, uri in VOICE_PRESETS.items()}
        self.info = BackendInfo("pocket_tts_farsi", "phonemes", True, presets)
        self.engine = Engine(model=config.model, g2p_model=config.g2p_model,
            model_config=config.model_config, reference_seconds=config.reference_seconds,
            output_dir=config.output_dir, retries=config.retries)

    def validate_config(self, config):
        if hasattr(self, "engine"):
            for name in ("model", "g2p_model", "model_config", "reference_seconds", "output_dir", "retries"):
                if getattr(config, name) != getattr(self.engine, name):
                    raise ValueError(f"Create a new Pocket backend when changing {name}")
        if config.backend_options:
            raise ValueError("pocket_tts_farsi uses the named Pocket settings; backend_options must be empty")
        if not config.model_config or not config.g2p_model:
            raise ValueError("Pocket Farsi needs model_config and g2p_model")

    @staticmethod
    def _require_runtime():
        from importlib.util import find_spec
        if any(find_spec(name) is None for name in ("pocket_tts", "torch", "transformers", "scipy")):
            raise ImportError('Install the Pocket runtime: uv sync --extra pocket or pip install "sokhanvar[pocket]"')

    def prepare(self, text, config):
        self._require_runtime()
        from ..api import Phrase, SpeechPlan
        normalized, rows = self.engine.prepare(text, config.comma_pause_ms,
                                               config.sentence_pause_ms, config.split_at_commas)
        return SpeechPlan(normalized, [Phrase(*row) for row in rows])

    def synthesize(self, plan, config):
        self._require_runtime()
        from ..api import SynthesisResult
        voice = config.reference_audio if "://" not in config.reference_audio else None
        wav, metadata, report = self.engine.render(plan.rows(), voice, config.seed,
            config.token_budget, config.temperature, "sokhanvar", config.eos_threshold,
            config.reference_audio, config.frames_after_eos)
        return SynthesisResult(Path(wav), Path(metadata), report)
