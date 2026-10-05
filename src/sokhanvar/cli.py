import argparse
from pathlib import Path
from . import Sokhanvar, SokhanvarConfig, SpeechPlan, available_backends


def main():
    parser = argparse.ArgumentParser(prog="sokhanvar")
    commands = parser.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("synthesize", help="Generate speech without Gradio")
    synth.add_argument("--config", required=True)
    source = synth.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    source.add_argument("--plan", help="Exported edited phrase plan YAML")
    synth.add_argument("--output", required=True)
    config = commands.add_parser("config", help="Write a default YAML config")
    config.add_argument("--output", default="sokhanvar.yaml")
    config.add_argument("--backend", default="pocket_tts_farsi")
    config.add_argument("--model", help="Model identifier; required for additional backends")
    commands.add_parser("backends", help="List built-in and installed backend plugins")
    playground = commands.add_parser("playground", help="Launch the optional Gradio UI")
    playground.add_argument("--config")
    playground.add_argument("--host", default="127.0.0.1")
    playground.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()
    if args.command == "config":
        SokhanvarConfig(backend=args.backend, model=args.model).to_yaml(args.output)
        print(args.output)
    elif args.command == "backends":
        print("\n".join(available_backends()))
    elif args.command == "playground":
        launch_playground(args.config, args.host, args.port)
    else:
        speaker = Sokhanvar.from_yaml(args.config)
        if args.plan:
            result = speaker.synthesize_plan(SpeechPlan.from_yaml(args.plan), args.output)
        else:
            text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else args.text
            result = speaker.synthesize(text, args.output)
        print(result.audio_path)


def launch_playground(config=None, host="127.0.0.1", port=7860):
    try:
        from .app import launch
    except ModuleNotFoundError as error:
        if error.name == "gradio":
            raise SystemExit('Install the playground extra: uv sync --extra playground or pip install "sokhanvar[playground]"') from error
        raise
    launch(config, host, port)
