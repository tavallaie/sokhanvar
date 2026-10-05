# Sokhanvar · سخنور

A Python library for Persian speech synthesis with YAML configuration and an optional Gradio playground. Repository: [tavallaie/sokhanvar](https://github.com/tavallaie/sokhanvar).

The built-in `pocket_tts_farsi` backend uses [Pocket TTS Farsi v2](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2) and its Persian G2P frontend. It preserves ezafe connections during chunking and lets you configure pauses, sampling, reference audio, and speech endings. Models load lazily and run on CPU.

## Install

From this repository, using uv:

```bash
uv sync                                      # Core library and CLI
uv sync --extra pocket                       # Pocket Farsi runtime
uv sync --extra pocket --extra playground    # Pocket Farsi and Gradio
```

In another project:

```bash
uv add "sokhanvar[pocket] @ git+https://github.com/tavallaie/sokhanvar.git"
# Or include the UI:
uv add "sokhanvar[pocket,playground] @ git+https://github.com/tavallaie/sokhanvar.git"
```

To use the current local checkout before it is pushed:

```bash
uv add --editable "/path/to/sokhanvar[pocket]"
```

This checkout configures uv to use CPU-only PyTorch. Other projects should also configure their PyTorch source if they want CPU-only wheels. For example, add PyTorch as a direct dependency with `uv add torch --index pytorch-cpu=https://download.pytorch.org/whl/cpu` before adding Sokhanvar. The package is not published to PyPI yet.

## Backend and model selection

Every backend targets Persian, with `language: fa`. YAML selects the backend adapter and the model separately. `backend_options` carries options specific to other adapters. `uv run sokhanvar backends` lists registered and installed adapters without loading their models.

Only `pocket_tts_farsi` ships with Sokhanvar today. Add other Persian TTS families through the [backend interface](docs/backends.md). A backend can consume Persian text directly or use its own pronunciation frontend. Pocket's G2P, ezafe token handling, and EOS settings stay in its adapter.

For Pocket models, omitting `model_config` derives `hf://{model}/model.yaml` from the selected model. An explicit HF config must belong to that same repository. When changing a config object with `dataclasses.replace`, pass `model_config=None` with the new model to derive the new URI. The replacement model must support this Persian Pocket adapter.

The playground displays the selected backend and model and exports both in YAML. Launch with a different config to switch models. Saved plans include backend and model identifiers; synthesis rejects a plan prepared for a different selection. Older plans without identifiers remain supported.

## Python API

```python
from sokhanvar import Sokhanvar, SokhanvarConfig

speaker = Sokhanvar(SokhanvarConfig(reference_audio="my-voice.wav"))
result = speaker.synthesize("کتاب دوست خوب من روی میز است.", "speech.wav")
print(result.audio_path)
print(result.metadata_path)
```

Or load settings exported by the playground:

```python
from sokhanvar import Sokhanvar

speaker = Sokhanvar.from_yaml("sokhanvar.yaml")
result = speaker.synthesize("امروز هوا خوب است. فردا به پارک می‌روم.", "speech.wav")
```

`SynthesisResult` contains `audio_path`, `metadata_path`, and a metadata dictionary. Reuse a `Sokhanvar` instance to reuse loaded models. Library imports and CLI synthesis do not import or require Gradio. For backends that require a reference, missing reference audio raises an error; the library never selects another voice silently.

To inspect or correct pronunciation before synthesis:

```python
plan = speaker.prepare("کتاب من روی میز است.")
print(plan.phrases[0].phonemes)
# Edit plan.phrases[i].phonemes or .pause_ms as needed.
plan.to_yaml("plan.yaml")
result = speaker.synthesize_plan(plan, "edited.wav")
```

Reload an edited plan with `SpeechPlan.from_yaml("plan.yaml")`. Using a plan preserves manually edited phonemes and per-phrase pauses instead of running G2P again.

## YAML configuration

See [examples/sokhanvar.yaml](examples/sokhanvar.yaml) for every setting. Place your reference clip beside it, or change `reference_audio`.

```yaml
backend: pocket_tts_farsi
language: fa
model: mehdi-hf/pocket-tts-farsi-v2
backend_options: {}
reference_audio: reference.wav
reference_seconds: 5.0
seed: 42
temperature: 0.3
eos_threshold: -2.0
frames_after_eos: 3
token_budget: 18
split_at_commas: false
comma_pause_ms: 150
sentence_pause_ms: 250
retries: 1
output_dir: outputs
```

Relative paths in a loaded YAML file resolve beside that file, including `reference_audio`, `output_dir`, and a local `model_config`. Unknown keys and invalid values raise errors. YAML loads use `safe_load`. Config objects are immutable; use `dataclasses.replace(config, seed=123)` to create changed settings.

The Pocket Farsi defaults use at most five seconds of the selected reference, an EOS threshold of −2, and three ending frames, about 240 ms. The original uploaded clip is preserved. A lower EOS threshold can stop speech early; a higher value can cause repetition. Zero ending frames can cut off a final sound.

## CLI without Gradio

```bash
uv run --extra pocket sokhanvar config --output sokhanvar.yaml
# Set reference_audio in the generated file, then:
uv run --extra pocket sokhanvar synthesize --config sokhanvar.yaml --text "سلام، حال شما چطور است؟" --output speech.wav
uv run --extra pocket sokhanvar synthesize --config sokhanvar.yaml --text-file article.txt --output article.wav
uv run --extra pocket sokhanvar synthesize --config sokhanvar.yaml --plan plan.yaml --output edited.wav
```

## Playground and config generator

```bash
uv run --extra pocket --extra playground sokhanvar playground
uv run --extra pocket --extra playground sokhanvar playground --config sokhanvar.yaml
```

Open http://127.0.0.1:7860. Use `--port` or `--host` to change the listening address. For Pocket Farsi, the first conversion downloads G2P weights; the first synthesis downloads TTS weights. Later runs reuse the Hugging Face cache.

Upload or record a short Persian reference. The playground saves it in `.cache/reference/` and restores it across page visits and restarts. Sample voices require explicit selection. Clearing the audio player forgets the saved selection. `PERSIAN_TTS_REFERENCE_DIR` changes the storage location.

Convert your text, edit A/B phonemes and pause columns, and compare the generated audio. Changing text or punctuation controls requires another conversion. Generation uses the phrase rows. For Pocket Farsi, edit the phoneme column to change pronunciation. A native text backend uses the editable Persian phrase column.

**Export YAML and portable bundle** downloads:

- `sokhanvar.yaml` with the current settings.
- `plan.yaml` with the selected A or B phrase plan, including manual edits.
- `sokhanvar.zip` containing both files and the original reference audio for local uploads.

Extract the ZIP in another project, then use `Sokhanvar.from_yaml` or the CLI. YAML alone does not contain reference audio. Remote reference URIs remain URIs. Custom local model configs and their weights must remain accessible at their configured paths.

## Persian pauses and ezafe

The built-in Pocket Farsi model consumes romanized phonemes. G2P drops punctuation, so Sokhanvar preserves selected boundaries before conversion and inserts silence between separately synthesized phrases. Complete sentences are the default; comma splitting is optional because short fragments can hurt pronunciation and rhythm. Question punctuation does not guarantee question intonation.

For کتابِ من, `ketAbe1 man` contains the spoken linking vowel `e`. The `1` protects the word connection during chunking and is removed before TTS. Compare with `ketAb man` to hear the difference. Adding or removing only `1` does not change the spoken vowel. Do not insert a comma inside an ezafe phrase.

The notation uses `A` for long آ, `a` for short a, `S` for ش, `C` for چ, `x` for خ, and `q` for ق/غ. `?` is a glottal stop and `;` is ژ, not punctuation pauses. G2P can misread words; the phrase editor is useful for corrections. Generated files and passing tests do not prove pronunciation or rhythm are correct.

Chunks are limited to 18 tokenizer tokens by default. Ezafe chains stay together; oversized chains raise an error. Extra chunk boundaries can affect rhythm even without inserted silence. Runaway or silent generations are retried according to `retries`, then reported as errors. This check is a heuristic, not a speech accuracy score.

## Model dependencies and licensing

The Pocket runtime is pinned to the PyPI release `pocket-tts==3.3.0`. This release supports the Persian model's capitalization flag, so no Git dependency is needed. Keep the model's YAML flags that disable capitalization, punctuation insertion, and short-input padding for phonemes. Synthesis uses the pinned runtime's short-text streaming method to bypass its orthographic splitter, where `?` otherwise means a question mark. The adapter supplies the cancellation event required by 3.3.0. Recheck that method before upgrading the runtime.

Use Transformers 5.15 or newer. Transformers 4 can load this G2P checkpoint incorrectly and produce repeated characters. The normalizer is copied from the model repository, retrieved 2026-10-05. No weights are bundled. Model repository IDs and generated phonemes appear in output JSON; model repositories currently use their default revisions.

The project uses the repository's [Apache 2.0 license](LICENSE). The TTS weights are CC BY-NC 4.0. See the model card for terms and attribution. The G2P repository includes its own license and attribution.

## Checks

```bash
uv run --extra pocket --extra playground python -m unittest discover -s tests -v
uv build
```

Tests cover configuration validation and portability, library imports without UI, phrase plans, reference selection, audio conditioning, pauses, and ezafe-safe chunking. Speech quality still requires listening.

## Publish to PyPI

Build release artifacts without uv-specific sources and check their metadata:

```bash
uv build --no-sources
uvx twine check dist/sokhanvar-0.1.0-py3-none-any.whl dist/sokhanvar-0.1.0.tar.gz
uv publish dist/sokhanvar-0.1.0-py3-none-any.whl dist/sokhanvar-0.1.0.tar.gz
```

Published dependencies, including extras, use package names and version constraints. PyPI rejects direct Git URL dependencies. Keep credentials out of project files; enter your PyPI token at the password prompt with username `__token__`.
