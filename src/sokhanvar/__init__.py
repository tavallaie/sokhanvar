"""Persian speech synthesis with optional Gradio playground."""
from .api import Phrase, Sokhanvar, SpeechPlan, SynthesisResult
from .config import SokhanvarConfig

__all__ = ["Sokhanvar", "SokhanvarConfig", "Phrase", "SpeechPlan", "SynthesisResult"]
