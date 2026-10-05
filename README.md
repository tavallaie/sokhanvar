# Sokhanvar

A Python library for Persian text-to-speech with YAML configuration and an optional Gradio playground.

The included backend uses [Pocket TTS Farsi v2](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2). Other Persian TTS models can integrate through [backend adapters](docs/backends.md).

## Installation

Requires Python 3.12, 3.13, or 3.14.

```bash
uv add "sokhanvar[pocket]"
```

Or with pip:

```bash
pip install "sokhanvar[pocket]"
```

For development from this repository:

```bash
uv sync --extra pocket --extra playground
```

The `pocket` extra installs the speech runtime. The `playground` extra installs Gradio and is optional.

## Usage

Save as `sokhanvar.yaml`:

```yaml
backend: pocket_tts_farsi
model: mehdi-hf/pocket-tts-farsi-v2
language: fa
reference_audio: reference.wav
output_dir: outputs
```

Place your reference recording beside the YAML file, then generate speech:

```python
from sokhanvar import Sokhanvar

speaker = Sokhanvar.from_yaml("sokhanvar.yaml")
result = speaker.synthesize("امروز هوا خوب است.", "speech.wav")
```

The result contains the WAV path, JSON metadata path, and generation metadata. Relative paths resolve beside the YAML file. Pocket Farsi uses up to five seconds of reference audio and downloads model weights on first use.

See [the full configuration](examples/sokhanvar.yaml) for pronunciation, pause, and sampling settings.

For CLI synthesis:

```bash
uv run sokhanvar synthesize \
  --config sokhanvar.yaml --text "امروز هوا خوب است." --output speech.wav
```

## Playground

```bash
uv add "sokhanvar[pocket,playground]"
uv run sokhanvar playground --config sokhanvar.yaml
```

Open [localhost:7860](http://127.0.0.1:7860) to compare A/B pronunciation and pauses. The playground exports YAML settings, edited phrase plans, and a ZIP containing the reference recording for use in another project.

## License

The library uses [Apache 2.0](LICENSE). Pocket TTS Farsi v2 weights use [CC BY-NC 4.0](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2). The [Persian pronunciation model](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF) has separate license terms.
