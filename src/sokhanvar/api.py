"""Public speech synthesis API with no UI imports."""
from dataclasses import dataclass
from pathlib import Path
import shutil
import yaml
from .config import SokhanvarConfig
from .engine import Engine


@dataclass
class Phrase:
    text: str
    phonemes: str
    pause_ms: int = 0


@dataclass
class SpeechPlan:
    normalized_text: str
    phrases: list[Phrase]

    def rows(self):
        return [[p.text, p.phonemes, p.pause_ms] for p in self.phrases]

    def to_yaml(self, path):
        from dataclasses import asdict
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def from_yaml(cls, path):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"normalized_text", "phrases"}:
            raise ValueError("Plan YAML needs normalized_text and phrases")
        if not isinstance(data["normalized_text"], str) or not isinstance(data["phrases"], list):
            raise ValueError("Invalid speech plan")
        return cls(data["normalized_text"], [Phrase(**row) for row in data["phrases"]])


@dataclass
class SynthesisResult:
    audio_path: Path
    metadata_path: Path
    metadata: dict


class Sokhanvar:
    def __init__(self, config: SokhanvarConfig | None = None):
        self.config = config or SokhanvarConfig()
        self.config.__post_init__()
        self._engine = Engine(model=self.config.model, g2p_model=self.config.g2p_model,
                              model_config=self.config.model_config,
                              reference_seconds=self.config.reference_seconds,
                              output_dir=self.config.output_dir, retries=self.config.retries)

    @classmethod
    def from_yaml(cls, path):
        return cls(SokhanvarConfig.from_yaml(path))

    def prepare(self, text: str) -> SpeechPlan:
        c = self.config
        normalized, rows = self._engine.prepare(text, c.comma_pause_ms, c.sentence_pause_ms, c.split_at_commas)
        return SpeechPlan(normalized, [Phrase(*row) for row in rows])

    def synthesize(self, text: str, output_path=None) -> SynthesisResult:
        self._check_reference()
        return self.synthesize_plan(self.prepare(text), output_path)

    def _check_reference(self):
        reference = self.config.reference_audio
        if not reference:
            raise ValueError("Set reference_audio in SokhanvarConfig. No fallback voice is selected automatically.")
        if "://" not in reference and not Path(reference).is_file():
            raise FileNotFoundError(reference)

    def synthesize_plan(self, plan: SpeechPlan, output_path=None) -> SynthesisResult:
        c = self.config
        c.__post_init__()
        self._check_reference()
        if output_path is not None and Path(output_path).suffix.lower() != ".wav":
            raise ValueError("output_path must end with .wav")
        voice = c.reference_audio if "://" not in c.reference_audio else None
        wav, metadata, report = self._engine.render(plan.rows(), voice, c.seed, c.token_budget,
            c.temperature, "sokhanvar", c.eos_threshold, c.reference_audio, c.frames_after_eos)
        audio_path, metadata_path = Path(wav), Path(metadata)
        if output_path is not None:
            target = Path(output_path)
            if target.suffix.lower() != ".wav":
                raise ValueError("output_path must end with .wav")
            target.parent.mkdir(parents=True, exist_ok=True)
            if audio_path.resolve() != target.resolve():
                shutil.copyfile(audio_path, target)
            target_metadata = target.with_suffix(".json")
            if metadata_path.resolve() != target_metadata.resolve():
                shutil.copyfile(metadata_path, target_metadata)
            audio_path, metadata_path = target, target_metadata
        return SynthesisResult(audio_path, metadata_path, report)
