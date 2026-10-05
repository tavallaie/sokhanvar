"""Contract for Persian speech backends. Importing it loads no model runtime."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Protocol

if TYPE_CHECKING:
    from ..api import SpeechPlan, SynthesisResult
    from ..config import SokhanvarConfig


@dataclass(frozen=True)
class BackendInfo:
    name: str
    input_kind: str = "text"
    requires_reference_audio: bool = False
    voice_presets: Mapping[str, str] = field(default_factory=dict)


class SpeechBackend(Protocol):
    info: BackendInfo

    def validate_config(self, config: "SokhanvarConfig") -> None:
        """Validate model compatibility and backend-specific options before loading weights."""
        ...

    def prepare(self, text: str, config: "SokhanvarConfig") -> "SpeechPlan":
        """Normalize Persian and prepare native text or phonemes for this backend."""
        ...

    def synthesize(self, plan: "SpeechPlan", config: "SokhanvarConfig") -> "SynthesisResult":
        """Write a WAV and return its path, a JSON sidecar path, and metadata."""
        ...
