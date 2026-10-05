"""Boundary planning independent of model inference."""
from dataclasses import dataclass
import re


@dataclass
class Segment:
    text: str
    pause_ms: int = 0


def split_persian(text: str, comma_ms: int = 150, sentence_ms: int = 250,
                  use_commas: bool = True) -> list[Segment]:
    """Preserve punctuation boundaries before G2P drops them."""
    parts = re.split(r"([.!؟?\n]+|[،,؛;:]+)", text.strip())
    result = []
    for index in range(0, len(parts), 2):
        body = parts[index].strip()
        mark = parts[index + 1] if index + 1 < len(parts) else ""
        if not body:
            continue
        sentence = bool(re.search(r"[.!؟?\n]", mark))
        if mark and not sentence and not use_commas:
            if index + 2 < len(parts):
                parts[index + 2] = body + " " + parts[index + 2].strip()
                continue
        result.append(Segment(body, sentence_ms if sentence else comma_ms if mark else 0))
    if result:
        result[-1].pause_ms = 0
    return result


def chunk_phonemes(text: str, count_tokens, budget: int = 18) -> list[str]:
    """Keep an ezafe chain together. Refuse a chain longer than the budget."""
    words = text.split()
    groups, group = [], []
    for word in words:
        group.append(word)
        if not word.endswith("1"):
            groups.append(" ".join(group))
            group = []
    if group:
        raise ValueError("An ezafe marker needs a following word. Add the word or remove the final 1.")
    chunks, current = [], ""
    for group in groups:
        if count_tokens(group.replace("1", "")) > budget:
            raise ValueError("A linked phrase exceeds the token budget. Shorten it or raise the budget.")
        candidate = f"{current} {group}".strip()
        if current and count_tokens(candidate.replace("1", "")) > budget:
            chunks.append(current)
            current = group
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def validate_phonemes(text: str) -> None:
    if not text.strip():
        raise ValueError("A phoneme row is empty. Convert the Persian text first.")
    if re.search(r"[^a-zA-Z?;1\s]", text):
        raise ValueError("Use romanized phonemes only. Persian script and punctuation must stay in the Persian editor. ? is a glottal stop; ; is zh.")
    for word in text.split():
        if not word.replace("1", ""):
            raise ValueError("An ezafe marker must follow a phoneme word.")
        if "1" in word and (not word.endswith("1") or word.count("1") != 1):
            raise ValueError("Place one ezafe marker 1 at the end of a word, after its spoken e or ye.")
