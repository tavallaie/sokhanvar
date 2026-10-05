"""Persian speech synthesis with optional Gradio playground."""
from .api import Phrase, Sokhanvar, SpeechPlan, SynthesisResult
from .config import SokhanvarConfig
from .backends import BackendInfo, SpeechBackend, available_backends, register_backend

__all__ = ["Sokhanvar", "SokhanvarConfig", "Phrase", "SpeechPlan", "SynthesisResult", "BackendInfo", "SpeechBackend", "available_backends", "register_backend"]
