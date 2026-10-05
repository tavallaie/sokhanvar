"""Run after placing reference.wav beside sokhanvar.yaml."""
from pathlib import Path
from sokhanvar import Sokhanvar

speaker = Sokhanvar.from_yaml(Path(__file__).with_name("sokhanvar.yaml"))
result = speaker.synthesize("کتاب دوست خوب من روی میز است.", "speech.wav")
print(result.audio_path)
