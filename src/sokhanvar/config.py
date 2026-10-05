"""Validated YAML settings. Relative paths resolve beside the YAML file."""
from dataclasses import asdict, dataclass, fields, field
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class SokhanvarConfig:
    backend: str = "pocket_tts_farsi"
    language: str = "fa"
    model: str | None = None
    g2p_model: str = "mehdi-hf/Homo-GE2PE-Persian-HF"
    model_config: str | None = None
    backend_options: dict = field(default_factory=dict)
    reference_audio: str | None = None
    reference_seconds: float = 5.0
    seed: int = 42
    temperature: float = 0.3
    eos_threshold: float = -2.0
    frames_after_eos: int = 3
    token_budget: int = 18
    split_at_commas: bool = False
    comma_pause_ms: int = 150
    sentence_pause_ms: int = 250
    retries: int = 1
    output_dir: str = "outputs"

    def __post_init__(self):
        if self.model is None and self.backend == "pocket_tts_farsi":
            object.__setattr__(self, "model", "mehdi-hf/pocket-tts-farsi-v2")
        for name in ("backend", "model", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.language != "fa":
            raise ValueError("Sokhanvar backends currently target Persian; language must be fa")
        if not isinstance(self.backend_options, dict) or any(not isinstance(k, str) for k in self.backend_options):
            raise ValueError("backend_options must be a mapping with string keys")
        if self.model_config is not None and not isinstance(self.model_config, str):
            raise ValueError("model_config must be a path, URI, or null")
        if self.backend == "pocket_tts_farsi":
            if self.model_config is None:
                object.__setattr__(self, "model_config", f"hf://{self.model}/model.yaml")
            for name in ("g2p_model", "model_config"):
                if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                    raise ValueError(f"{name} must be a nonempty string for Pocket Farsi")
            if self.model_config.startswith("hf://"):
                repo = self.model_config[5:].split("/")[:2]
                if "/".join(repo) != self.model:
                    raise ValueError("model_config must point to the selected model repository, not a different model")
        if self.reference_audio is not None and (not isinstance(self.reference_audio, str) or not self.reference_audio.strip()):
            raise ValueError("reference_audio must be a path, URI, or null")
        if type(self.split_at_commas) is not bool:
            raise ValueError("split_at_commas must be true or false")
        limits = {"seed": (0, 2147483647), "comma_pause_ms": (0, 2000),
                  "sentence_pause_ms": (0, 2000), "retries": (0, 5)}
        if self.backend == "pocket_tts_farsi":
            limits.update(frames_after_eos=(0, 8), token_budget=(9, 18))
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")
        ranges = (("temperature", 0.1, 1.2), ("eos_threshold", -6, 0), ("reference_seconds", 0.1, 5)) if self.backend == "pocket_tts_farsi" else ()
        for name, low, high in ranges:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")

    @classmethod
    def from_yaml(cls, path):
        path = Path(path).resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config YAML must contain a mapping")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("Unknown config settings: " + ", ".join(sorted(map(str, unknown))))
        validated = cls(**data)
        for name in ("reference_audio", "output_dir", "model_config"):
            value = getattr(validated, name)
            if value and "://" not in value and not Path(value).is_absolute():
                data[name] = str((path.parent / value).resolve())
        return cls(**data)

    def to_yaml(self, path):
        # Validate again in case a caller edited the dataclass fields.
        self.__post_init__()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        for name in ("reference_audio", "output_dir", "model_config"):
            value = data[name]
            if value and "://" not in value and not Path(value).is_absolute():
                data[name] = str(Path(value).resolve())
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path
