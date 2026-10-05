"""Explicit, lazy backend lookup. YAML never imports arbitrary Python paths."""
from importlib.metadata import entry_points


def _pocket(config):
    from .pocket import PocketFarsiBackend
    return PocketFarsiBackend(config)


_factories = {"pocket_tts_farsi": _pocket}


def register_backend(name, factory, *, replace=False):
    if not isinstance(name, str) or not name.strip() or not callable(factory):
        raise ValueError("Register a nonempty backend name and callable factory")
    if not replace and name in available_backends():
        raise ValueError(f"Backend {name!r} is already registered")
    _factories[name] = factory


def _plugins():
    return entry_points(group="sokhanvar.backends")


def available_backends():
    return sorted(set(_factories) | {plugin.name for plugin in _plugins()})


def create_backend(config):
    factory = _factories.get(config.backend)
    if factory is None:
        plugins = [p for p in _plugins() if p.name == config.backend]
        if len(plugins) > 1:
            raise ValueError(f"Multiple plugins register backend {config.backend!r}")
        if plugins:
            factory = plugins[0].load()
    if factory is None:
        raise ValueError(f"Unknown backend {config.backend!r}. Available backends: {', '.join(available_backends())}")
    backend = factory(config)
    if backend.info.name != config.backend:
        raise ValueError("Backend factory returned a different backend name")
    if backend.info.input_kind not in ("text", "phonemes"):
        raise ValueError("Backend input_kind must be text or phonemes")
    backend.validate_config(config)
    return backend
