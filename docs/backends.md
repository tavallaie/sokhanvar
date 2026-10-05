# Add a Persian TTS backend

The core package depends on PyYAML. Install `sokhanvar[pocket]` only when using Pocket Farsi, and `sokhanvar[playground]` when using Gradio. Each additional backend supplies its own runtime dependencies.

Implement the `SpeechBackend` protocol exported by `sokhanvar`. Its methods receive the selected `SokhanvarConfig`:

| Member | Contract |
| --- | --- |
| `info` | `BackendInfo(name, input_kind="text", requires_reference_audio=False, voice_presets={})`. Input kind is `text` or `phonemes`. |
| `__init__(config)` | Store the model selection and initialization settings. Prefer lazy model loading. |
| `validate_config(config)` | Validate `backend_options` and any settings the adapter supports. Reject unsupported options. |
| `prepare(text, config)` | Accept Persian text and return `SpeechPlan(normalized_text, phrases)`. Each `Phrase` has `text`, optional `phonemes`, and `pause_ms`. |
| `synthesize(plan, config)` | Generate the edited phrases, apply their pauses, and return `SynthesisResult(audio_path, metadata_path, metadata)`. Return WAV audio and reserve a JSON metadata path. |

The facade stamps backend and model identifiers on prepared plans, checks required reference audio, and writes metadata with the selected backend, model, and language. The adapter owns normalization, pronunciation, model inference, audio format conversion, and insertion of silence. A text backend reads `Phrase.text`; it must not apply Pocket's phoneme translation. Document any pronunciation overrides your backend supports.

The seed and pause settings are shared configuration. `g2p_model`, `model_config`, `token_budget`, `temperature`, `eos_threshold`, and `frames_after_eos` are legacy Pocket settings. Other adapters should put model-specific options in `backend_options` and validate them themselves. Pocket's numeric limits do not constrain other adapters.

Register an adapter in an application before loading YAML:

```python
from sokhanvar import Sokhanvar, register_backend
from my_project.persian_backend import MyPersianBackend

register_backend("my_persian_tts", MyPersianBackend)
speaker = Sokhanvar.from_yaml("speech.yaml")
result = speaker.synthesize("کتاب دوست من روی میز است.", "speech.wav")
```

`MyPersianBackend` is your implementation of the protocol above. Its `info.name` must match the registered name. Reusing a `Sokhanvar` instance reuses the adapter and its loaded models.

For CLI and playground discovery, publish a Python package with this entry point in its `pyproject.toml`:

```toml
[project.entry-points."sokhanvar.backends"]
my_persian_tts = "my_project.persian_backend:MyPersianBackend"
```

Install that package in the same environment as Sokhanvar. Discovery lists names without importing plugins; choosing a backend loads its factory. YAML never imports arbitrary Python module paths. Unknown backend names raise an error.

The following config is a template for your adapter, not a second bundled model:

```yaml
backend: my_persian_tts
language: fa
model: your-organization/your-persian-model
backend_options:
  speed: 1.0
reference_audio: null
output_dir: outputs
```

The backend must support the selected model and `speed` option. Model identifiers may be repository IDs or local paths as defined by the adapter. Paths inside `backend_options` are adapter-owned and are not automatically resolved or copied into portable exports. Reference audio and the core output paths retain their usual YAML-relative behavior.

Launch with `sokhanvar playground --config speech.yaml` to use the same adapter in Gradio. The playground shows the selected backend and model, uses its voice presets, hides Pocket controls for other adapters, and exports `backend_options` unchanged. Model-specific options are edited in YAML.
