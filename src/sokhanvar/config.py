"""Validated YAML settings. Relative paths resolve beside the YAML file."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class SokhanvarConfig:
    model: str = "mehdi-hf/pocket-tts-farsi-v2"
    g2p_model: str = "mehdi-hf/Homo-GE2PE-Persian-HF"
    model_config: str = "hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml"
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
        for name in ("model", "g2p_model", "model_config", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.reference_audio is not None and (not isinstance(self.reference_audio, str) or not self.reference_audio.strip()):
            raise ValueError("reference_audio must be a path, URI, or null")
        if type(self.split_at_commas) is not bool:
            raise ValueError("split_at_commas must be true or false")
        limits = {"seed": (0, 2147483647), "frames_after_eos": (0, 8),
                  "token_budget": (9, 18), "comma_pause_ms": (0, 2000),
                  "sentence_pause_ms": (0, 2000), "retries": (0, 5)}
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")
        for name, low, high in (("temperature", 0.1, 1.2), ("eos_threshold", -6, 0), ("reference_seconds", 0.1, 5)):
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
        cls(**data)
        for name in ("reference_audio", "output_dir", "model_config"):
            value = data.get(name, getattr(cls(), name))
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
