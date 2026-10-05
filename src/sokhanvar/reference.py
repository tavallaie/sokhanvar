"""Persist the user's reference and require an explicit choice of sample voices."""
import hashlib
import json
import os
from pathlib import Path
import shutil

UPLOAD_CHOICE = "My uploaded reference"
REFERENCE_DIR = Path(os.environ.get("PERSIAN_TTS_REFERENCE_DIR", ".cache/reference"))


def saved_reference():
    manifest = REFERENCE_DIR / "selected.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    path = Path(data["path"])
    return str(path.resolve()) if path.is_file() else None


def persist_reference(path):
    if not path:
        raise ValueError("Upload or record your reference audio first.")
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    target = REFERENCE_DIR / f"{digest}{source.suffix}"
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    manifest = {"path": str(target.resolve()), "filename": source.name, "sha256": digest}
    (REFERENCE_DIR / "selected.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(target.resolve())


def clear_reference():
    (REFERENCE_DIR / "selected.json").unlink(missing_ok=True)


def resolve_reference(path, choice, presets):
    if choice == UPLOAD_CHOICE:
        if not path:
            raise ValueError("Your reference audio is missing. Upload or record it before generating, or explicitly select a sample voice.")
        return persist_reference(path), presets[next(iter(presets))]
    return None, presets[choice]
