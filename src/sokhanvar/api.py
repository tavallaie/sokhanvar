"""Public speech synthesis API with no UI imports."""
from dataclasses import dataclass
from pathlib import Path
import shutil
import json
import yaml
from .config import SokhanvarConfig
from .backends.registry import create_backend


@dataclass
class Phrase:
    text: str
    phonemes: str = ""
    pause_ms: int = 0


@dataclass
class SpeechPlan:
    normalized_text: str
    phrases: list[Phrase]
    backend: str | None = None
    model: str | None = None

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
        if not isinstance(data, dict) or not {"normalized_text", "phrases"} <= set(data) or set(data) - {"normalized_text", "phrases", "backend", "model"}:
            raise ValueError("Plan YAML needs normalized_text and phrases")
        if not isinstance(data["normalized_text"], str) or not isinstance(data["phrases"], list):
            raise ValueError("Invalid speech plan")
        return cls(data["normalized_text"], [Phrase(**row) for row in data["phrases"]], data.get("backend"), data.get("model"))


@dataclass
class SynthesisResult:
    audio_path: Path
    metadata_path: Path
    metadata: dict


class Sokhanvar:
    def __init__(self, config: SokhanvarConfig | None = None, *, backend=None):
        self.config = config or SokhanvarConfig()
        self.config.__post_init__()
        self.backend = backend if backend is not None else create_backend(self.config)
        if self.backend.info.name != self.config.backend:
            raise ValueError("Backend instance does not match the selected backend")
        self.backend.validate_config(self.config)

    @classmethod
    def from_yaml(cls, path):
        return cls(SokhanvarConfig.from_yaml(path))

    def prepare(self, text: str) -> SpeechPlan:
        plan = self.backend.prepare(text, self.config)
        plan.backend, plan.model = self.config.backend, self.config.model
        return plan

    def synthesize(self, text: str, output_path=None) -> SynthesisResult:
        self._check_reference()
        return self.synthesize_plan(self.prepare(text), output_path)

    def _check_reference(self):
        reference = self.config.reference_audio
        if not reference and self.backend.info.requires_reference_audio:
            raise ValueError("Set reference_audio in SokhanvarConfig. No fallback voice is selected automatically.")
        if reference and "://" not in reference and not Path(reference).is_file():
            raise FileNotFoundError(reference)

    def synthesize_plan(self, plan: SpeechPlan, output_path=None) -> SynthesisResult:
        c = self.config
        c.__post_init__()
        self._check_reference()
        if plan.backend and plan.backend != c.backend:
            raise ValueError("This speech plan belongs to another backend. Prepare it with the selected backend.")
        if plan.model and plan.model != c.model:
            raise ValueError("This speech plan belongs to another model. Prepare it with the selected model.")
        if output_path is not None and Path(output_path).suffix.lower() != ".wav":
            raise ValueError("output_path must end with .wav")
        result = self.backend.synthesize(plan, c)
        report = dict(result.metadata, backend=c.backend, model=c.model, language=c.language)
        audio_path, metadata_path = Path(result.audio_path), Path(result.metadata_path)
        metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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
