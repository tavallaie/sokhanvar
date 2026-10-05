"""Lazy CPU inference. One lock protects both RNG and model state."""
import json
import hashlib
import os
from pathlib import Path
import threading
import time
import uuid

from .normalize_fa import normalize
from .prosody import chunk_phonemes, split_persian, validate_phonemes

MODEL = "mehdi-hf/pocket-tts-farsi-v2"
G2P = "mehdi-hf/Homo-GE2PE-Persian-HF"
CONFIG = f"hf://{MODEL}/model.yaml"
DEFAULT_VOICE = f"hf://{MODEL}/samples/prompt_short_sentence.wav"
VOICE_PRESETS = {
    "Persian female · model sample": DEFAULT_VOICE,
    "Persian male · model sample": f"hf://{MODEL}/samples/prompt_news_paragraph.wav",
    "English Alba · comparison": "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
}
TRANSLATE = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})
LOCK = threading.RLock()


class Engine:
    def __init__(self, model=MODEL, g2p_model=G2P, model_config=CONFIG,
                 reference_seconds=5.0, output_dir=None, retries=1):
        self.tokenizer = self.g2p = self.tts = None
        self.model, self.g2p_model, self.model_config = model, g2p_model, model_config
        self.reference_seconds, self.output_dir, self.retries = reference_seconds, output_dir, retries

    def phonemise(self, text):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        if self.g2p is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.g2p_model)
            self.g2p = T5ForConditionalGeneration.from_pretrained(self.g2p_model).eval()
        encoded = self.tokenizer([text.replace("؟", "").replace("?", "")],
                                 add_special_tokens=False, return_tensors="pt")
        if encoded.input_ids.shape[-1] > 450:
            raise ValueError("This phrase is too long for G2P. Insert a sentence or clause boundary.")
        with torch.inference_mode():
            output = self.g2p.generate(**encoded, num_beams=5, max_length=512, early_stopping=True)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip().translate(TRANSLATE)

    def prepare(self, text, comma_ms, sentence_ms, use_commas):
        if not text.strip() or len(text) > 4000:
            raise ValueError("Enter between 1 and 4000 characters of Persian text.")
        if any(c.isascii() and c.isalpha() for c in text):
            raise ValueError("Write Latin names in Persian here, or edit their phonemes after conversion.")
        normalized = normalize(text)
        segments = split_persian(normalized, int(comma_ms), int(sentence_ms), use_commas)
        if not segments:
            raise ValueError("No Persian text remains after normalization.")
        with LOCK:
            rows = [[s.text, self.phonemise(s.text), s.pause_ms] for s in segments]
        for row in rows:
            validate_phonemes(row[1])
        return normalized, rows

    def load_tts(self):
        if self.tts is None:
            from pocket_tts import TTSModel
            self.tts = TTSModel.load_model(config=self.model_config, eos_threshold=-2.0)
            for flag in ("capitalize_first_letter", "append_terminal_punctuation", "pad_with_spaces_for_short_inputs"):
                if getattr(self.tts, flag, True):
                    self.tts = None
                    raise RuntimeError("The model config must disable " + flag)

    def render(self, rows, voice, seed, budget, temperature, label,
               eos_threshold=-2.0, voice_preset=DEFAULT_VOICE, frames_after_eos=3):
        import numpy as np
        import torch
        from scipy.io.wavfile import write
        if hasattr(rows, "values"):
            rows = rows.values.tolist()
        if not rows or len(rows) > 100:
            raise ValueError("Prepare 1 to 100 phrase rows before generating.")
        cleaned = []
        for row in rows:
            phonemes = str(row[1]).strip()
            validate_phonemes(phonemes)
            pause = float(row[2])
            if not np.isfinite(pause) or not 0 <= pause <= 2000:
                raise ValueError("Each pause must be between 0 and 2000 ms.")
            cleaned.append((str(row[0]), phonemes, int(pause)))
        started = time.monotonic()
        with LOCK:
            self.load_tts()
            self.tts.temp = float(temperature)
            self.tts.eos_threshold = float(eos_threshold)
            torch.manual_seed(int(seed))
            reference = voice or voice_preset
            conditioning = reference
            reference_seconds = None
            local_reference = reference
            if "://" in str(reference):
                from pocket_tts.utils.utils import download_if_necessary
                local_reference = download_if_necessary(reference)
            if Path(local_reference).is_file():
                from pocket_tts.data.audio import audio_read
                from pocket_tts.data.audio_utils import convert_audio
                reference_audio, reference_rate = audio_read(local_reference)
                reference_seconds = reference_audio.shape[-1] / reference_rate
                # Persian training capped reference prompts at five seconds.
                reference_audio = reference_audio[..., :round(self.reference_seconds * reference_rate)]
                conditioning = convert_audio(reference_audio, reference_rate, self.tts.sample_rate, 1)
            voice_state = self.tts.get_state_for_audio_prompt(conditioning)
            tokenizer = self.tts.flow_lm.conditioner.tokenizer
            def count(text):
                ids = tokenizer.sp.encode(text, out_type=int)
                if tokenizer.sp.unk_id() in ids:
                    raise ValueError("Phonemes contain a symbol outside this model's vocabulary. Check capitalization and the phoneme notation.")
                return len(ids)
            pieces, report = [], []
            for row_index, (persian, phonemes, pause) in enumerate(cleaned):
                chunks = chunk_phonemes(phonemes, count, int(budget))
                for chunk_index, chunk in enumerate(chunks):
                    spoken = chunk.replace("1", "")
                    tokens = count(spoken)
                    cap = tokens / 3.0 + 2.0
                    for attempt in range(self.retries + 1):
                        # Already chunked phonemes must bypass the orthographic
                        # splitter: ? is a glottal stop, not a question mark.
                        stream = self.tts._generate_audio_stream_short_text(
                            model_state=voice_state, text_to_generate=spoken,
                            frames_after_eos=int(frames_after_eos),
                            copy_state=True,
                        )
                        audio = torch.cat(list(stream)).detach().cpu().numpy().reshape(-1)
                        duration = len(audio) / self.tts.sample_rate
                        audible = np.isfinite(audio).all() and np.max(np.abs(audio), initial=0) > 1e-5
                        if duration < cap * 0.98 and duration > 0.08 and audible:
                            break
                    else:
                        raise RuntimeError(f"Generation reached its length cap or returned silence after {self.retries + 1} attempts. Shorten the phrase or try another seed.")
                    pieces.append(audio)
                    last_chunk = chunk_index == len(chunks) - 1
                    gap = pause if last_chunk and row_index < len(cleaned) - 1 else 0
                    pieces.append(np.zeros(round(self.tts.sample_rate * gap / 1000), dtype=np.float32))
                    report.append({"persian": persian, "phonemes_with_ezafe": chunk,
                                   "model_input": spoken, "tokens": tokens,
                                   "duration_seconds": round(duration, 3), "inserted_pause_ms": gap,
                                   "attempts": attempt + 1,
                                   "note": "Short chunk. Compare with comma splitting disabled or a longer phrase." if tokens < 9 else ""})
            waveform = np.concatenate(pieces)
            sample_rate = self.tts.sample_rate
        folder = Path(self.output_dir or os.environ.get("PERSIAN_TTS_OUTPUT_DIR", "outputs"))
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f"{label}-{uuid.uuid4().hex[:12]}"
        audio_path, metadata_path = stem.with_suffix(".wav"), stem.with_suffix(".json")
        write(audio_path, sample_rate, (np.clip(waveform, -1, 1) * 32767).astype(np.int16))
        metadata = {"model": self.model, "g2p": self.g2p_model, "config": self.model_config, "seed": int(seed),
                    "voice": "uploaded reference" if voice else reference,
                    "reference_filename": Path(voice).name if voice else None,
                    "reference_sha256": hashlib.sha256(Path(voice).read_bytes()).hexdigest() if voice and Path(voice).is_file() else None,
                    "reference_duration_seconds": reference_seconds,
                    "reference_used_seconds": min(reference_seconds, self.reference_seconds) if reference_seconds is not None else None,
                    "eos_threshold": float(eos_threshold), "frames_after_eos": int(frames_after_eos),
                    "temperature": temperature, "token_budget": int(budget), "chunks": report,
                    "elapsed_seconds": round(time.monotonic() - started, 2)}
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(audio_path), str(metadata_path), metadata
