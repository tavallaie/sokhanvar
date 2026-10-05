"""Local Gradio playground."""
import argparse
import functools
import gradio as gr

from .engine import VOICE_PRESETS
from .reference import UPLOAD_CHOICE, saved_reference, persist_reference, clear_reference, resolve_reference
from .config import SokhanvarConfig
from .api import Phrase, SpeechPlan
from .engine import Engine
from .export import export_bundle

EXAMPLES = [
    ["Ezafe chain", "کتاب دوست خوب من روی میز است."],
    ["Comma and sentence pauses", "امروز هوا خوب است، اما من در خانه می‌مانم. فردا به پارک می‌روم."],
    ["A pause changes the meaning", "بخشش، لازم نیست اعدامش کنید."],
    ["The other reading", "بخشش لازم نیست، اعدامش کنید."],
    ["Numbers", "تا سال ۲۰۳۰، جمعیت شهر تغییر می‌کند."],
    ["Question", "این کتاب برای شماست؟"],
]
CSS = """
.gradio-container {max-width: 1280px !important;}
#persian-input textarea, #normalized textarea {direction: rtl; text-align: right; font-size: 20px; line-height: 1.9;}
#intro {padding: 20px 24px; border-radius: 16px; background: linear-gradient(110deg,#133d44,#205e56); color: white; margin-bottom: 18px;}
#intro h1, #intro p {color: white;}
"""


def friendly(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as error:
            raise gr.Error(str(error)) from error
    return wrapped


def build_app(config=None):
    config = config or SokhanvarConfig()
    engine = Engine(model=config.model, g2p_model=config.g2p_model,
                    model_config=config.model_config, reference_seconds=config.reference_seconds,
                    output_dir=config.output_dir, retries=config.retries)
    with gr.Blocks(title="Sokhanvar") as app:
        gr.Markdown("# Sokhanvar · سخنور\nPersian speech playground and YAML config generator.", elem_id="intro")
        gr.Markdown("Write Persian, convert it to phonemes, then edit two versions and listen. The first conversion downloads G2P weights; the first generation downloads speech weights. Inference runs locally on CPU.")
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Persian text · متن فارسی", value=EXAMPLES[0][1], lines=4, elem_id="persian-input")
                gr.Examples(examples=[[e[1]] for e in EXAMPLES], inputs=[text], label="Test sentences")
            with gr.Column(scale=2):
                configured_local = config.reference_audio and "://" not in config.reference_audio
                saved = config.reference_audio if configured_local else saved_reference()
                voice = gr.Audio(value=saved, label="Your reference voice · صدای مرجع شما", sources=["upload", "microphone"], type="filepath")
                gr.Markdown("Your uploaded reference is saved locally and restored after refreshes and restarts. Sample voices are used only when you select one.")
                presets = dict(VOICE_PRESETS)
                if config.reference_audio and not configured_local:
                    presets["Reference from config"] = config.reference_audio
                initial_choice = "Reference from config" if config.reference_audio and not configured_local else UPLOAD_CHOICE
                preset = gr.Dropdown(choices=[UPLOAD_CHOICE, *presets], value=initial_choice, label="Voice to use for generation")
                voice_status = gr.Markdown("Using your saved reference audio." if saved else "Upload or record your reference audio before generating.")
                use_commas = gr.Checkbox(value=config.split_at_commas, label="Split at commas and clause punctuation")
                gr.Markdown("Start with complete sentences. Enable comma splitting to test a pause; short fragments can hurt pronunciation and rhythm.")
                comma = gr.Slider(0, 2000, value=config.comma_pause_ms, step=10, label="Clause pause · مکث کوتاه · ms")
                sentence = gr.Slider(0, 2000, value=config.sentence_pause_ms, step=10, label="Sentence pause · مکث جمله · ms")
        prepare = gr.Button("Convert and prepare A / B", variant="primary")
        normalized = gr.Textbox(label="Normalized Persian · متن نرمال‌شده", interactive=False, elem_id="normalized")
        gr.Markdown("### Edit the pronunciation and pauses\nEach row is one phrase. The final column adds silence after that phrase. The last row has no trailing pause. Changing the text above requires another conversion; generation uses the rows below.\n\n**Ezafe:** `ketAbe1 man` keeps کتابِ من together. The spoken `e` is already in `ketAbe`; `1` only protects the link and is removed before synthesis. To test without ezafe, compare `ketAb man`. Do not add a comma between words that belong to one ezafe phrase.\n\n`A` = long آ, `a` = short a, `S` = ش, `C` = چ, `x` = خ, `q` = ق/غ. In phonemes, `?` is a glottal stop and `;` is ژ, so neither is a punctuation pause.")
        with gr.Row():
            seed = gr.Number(value=config.seed, minimum=0, maximum=2147483647, precision=0, label="Seed shared by A and B")
            budget = gr.Slider(9, 18, value=config.token_budget, step=1, label="Maximum tokenizer tokens per chunk")
            temperature = gr.Slider(0.1, 1.2, value=config.temperature, step=0.05, label="Sampling temperature")
        eos = gr.Slider(-6, 0, value=config.eos_threshold, step=0.25, label="End-of-speech threshold · Persian default −2")
        tail = gr.Slider(0, 8, value=config.frames_after_eos, step=1, label="Audio frames after end-of-speech · default 3 ≈ 240 ms")
        gr.Markdown("The app uses at most the first five seconds of your reference, matching Persian training. A short audio tail helps preserve final sounds. Missing whole words may need a higher EOS threshold or another seed.")
        gr.Markdown("A lower threshold can stop speech too early. A higher threshold can run longer or repeat words. The Persian synthesis script uses −2.")
        gr.Markdown("Long phrases may require extra chunks. The app avoids splitting at `1` links, but those extra synthesis boundaries can still affect rhythm. Insert a clause boundary yourself if the result sounds unnatural. A fixed seed reduces sampling differences; changing chunk boundaries also changes the generated speech.")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Version A")
                rows_a = gr.Dataframe(headers=["Persian phrase", "Phonemes · editable", "Pause after · ms"], datatype=["str", "str", "number"], type="array", interactive=True, column_count=(3, "fixed"), label="A phrase plan")
                run_a = gr.Button("Generate A", variant="primary")
                audio_a = gr.Audio(label="Listen to A", type="filepath")
                file_a = gr.File(label="A settings and exact model input")
            with gr.Column():
                gr.Markdown("### Version B")
                rows_b = gr.Dataframe(headers=["Persian phrase", "Phonemes · editable", "Pause after · ms"], datatype=["str", "str", "number"], type="array", interactive=True, column_count=(3, "fixed"), label="B phrase plan")
                run_b = gr.Button("Generate B", variant="primary")
                audio_b = gr.Audio(label="Listen to B", type="filepath")
                file_b = gr.File(label="B settings and exact model input")
        both = gr.Button("Generate both versions")
        with gr.Accordion("Export for another project", open=True):
            export_version = gr.Radio(["A", "B"], value="A", label="Phrase plan to export")
            export_button = gr.Button("Export YAML and portable bundle")
            gr.Markdown("The ZIP includes YAML settings, your reference clip, and the selected edited phrase plan. Extract it in another project. YAML alone does not contain audio. Export uses current controls; reconvert first if you changed punctuation settings.")
            exported_config = gr.File(label="Settings YAML")
            exported_plan = gr.File(label="Edited phrase plan YAML")
            exported_bundle = gr.File(label="Portable ZIP with reference audio")
        with gr.Accordion("Inspect generation details", open=False):
            with gr.Row():
                report_a = gr.JSON(label="A chunks")
                report_b = gr.JSON(label="B chunks")
        gr.Markdown("### What to listen for\nDoes the vowel linking related words sound right? Does a pause interrupt that link? Are clause and sentence breaks clear? Is the first word complete? Try a few seeds before deciding one version is better.\n\nPunctuation marks define boundaries here; the app inserts the selected amount of silence between separate generations. This does not guarantee question intonation or a particular delivery. Ezafe decisions come from G2P and may need correction by a Persian speaker.\n\n[Model and noncommercial license](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2) · [Persian G2P](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF)")

        @friendly
        def convert(*args):
            norm, rows = engine.prepare(*args)
            return norm, rows, [row.copy() for row in rows], None, None, None, None, {}, {}

        @friendly
        def generate(rows, reference, seed_value, token_budget, temp, eos_value, preset_name, tail_value, label):
            selected, sample = resolve_reference(reference, preset_name, presets)
            return engine.render(rows, selected, seed_value, token_budget, temp, label,
                                 eos_value, sample, tail_value)

        @friendly
        def generate_both(a, b, reference, seed_value, token_budget, temp, eos_value, preset_name, tail_value):
            selected, sample = resolve_reference(reference, preset_name, presets)
            return (*engine.render(a, selected, seed_value, token_budget, temp, "A", eos_value, sample, tail_value),
                    *engine.render(b, selected, seed_value, token_budget, temp, "B", eos_value, sample, tail_value))

        @friendly
        def export_settings(a, b, version, normalized_text, reference, preset_name,
                            seed_value, token_budget, temp, eos_value, tail_value,
                            comma_value, sentence_value, split_commas):
            from dataclasses import replace
            selected, sample = resolve_reference(reference, preset_name, presets)
            settings = replace(config, reference_audio=selected or sample,
                seed=int(seed_value), token_budget=int(token_budget), temperature=float(temp),
                eos_threshold=float(eos_value), frames_after_eos=int(tail_value),
                comma_pause_ms=int(comma_value), sentence_pause_ms=int(sentence_value),
                split_at_commas=split_commas)
            rows = a if version == "A" else b
            if not rows:
                raise ValueError("Convert your text before exporting a phrase plan.")
            plan = SpeechPlan(normalized_text, [Phrase(str(r[0]), str(r[1]), int(r[2])) for r in rows])
            return export_bundle(settings, plan)

        export_button.click(export_settings,
            [rows_a, rows_b, export_version, normalized, voice, preset, seed, budget,
             temperature, eos, tail, comma, sentence, use_commas],
            [exported_config, exported_plan, exported_bundle])

        @friendly
        def keep_upload(path):
            persist_reference(path)
            return UPLOAD_CHOICE, "Using your uploaded reference audio. Saved locally for the next visit."

        def forget_upload():
            clear_reference()
            return "Your reference was cleared. Upload or record it before generating with your voice."

        def show_reference(choice, path):
            if choice == UPLOAD_CHOICE:
                return "Using your uploaded reference audio." if path else "Your reference is missing. Upload or record it before generating."
            return "Using sample reference: " + choice

        def restore_reference():
            path = config.reference_audio if configured_local else saved_reference()
            return path, initial_choice, show_reference(initial_choice, path)

        voice.upload(keep_upload, [voice], [preset, voice_status])
        voice.stop_recording(keep_upload, [voice], [preset, voice_status])
        voice.clear(forget_upload, [], [voice_status])
        preset.change(show_reference, [preset, voice], [voice_status])
        app.load(restore_reference, [], [voice, preset, voice_status])

        prepare.click(convert, [text, comma, sentence, use_commas],
                      [normalized, rows_a, rows_b, audio_a, audio_b, file_a, file_b, report_a, report_b])
        run_a.click(functools.partial(generate, label="A"), [rows_a, voice, seed, budget, temperature, eos, preset, tail], [audio_a, file_a, report_a])
        run_b.click(functools.partial(generate, label="B"), [rows_b, voice, seed, budget, temperature, eos, preset, tail], [audio_b, file_b, report_b])
        both.click(generate_both, [rows_a, rows_b, voice, seed, budget, temperature, eos, preset, tail],
                   [audio_a, file_a, report_a, audio_b, file_b, report_b])
    return app


def main():
    parser = argparse.ArgumentParser(description="Sokhanvar playground and YAML config generator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--config")
    args = parser.parse_args()
    launch(args.config, args.host, args.port)


def launch(config_path=None, host="127.0.0.1", port=7860):
    config = SokhanvarConfig.from_yaml(config_path) if config_path else SokhanvarConfig()
    build_app(config).queue(default_concurrency_limit=1).launch(server_name=host, server_port=port, css=CSS,
                                                       theme=gr.themes.Soft(primary_hue="teal"))


if __name__ == "__main__":
    main()
