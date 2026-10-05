"""Portable config, reference audio, and editable phrase plan export."""
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import zipfile
from .config import SokhanvarConfig
from .api import SpeechPlan


def export_bundle(config: SokhanvarConfig, plan: SpeechPlan):
    folder = Path(tempfile.mkdtemp(prefix="sokhanvar-export-"))
    reference = config.reference_audio
    if reference and "://" not in reference:
        source = Path(reference)
        if not source.is_file():
            raise FileNotFoundError(source)
        name = "reference" + source.suffix
        shutil.copyfile(source, folder / name)
        reference = name
    # A relative reference intentionally belongs next to this exported YAML.
    import yaml
    from dataclasses import asdict
    data = asdict(replace(config, reference_audio=reference, output_dir="outputs"))
    yaml_path = folder / "sokhanvar.yaml"
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    plan_path = plan.to_yaml(folder / "plan.yaml")
    bundle = folder / "sokhanvar.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in folder.iterdir():
            if path != bundle:
                archive.write(path, path.name)
    return str(yaml_path), str(plan_path), str(bundle)
