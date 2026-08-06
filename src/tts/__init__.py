"""TTS engine registry — add new backends without touching existing code."""

import importlib
from typing import Dict, Type

from .base import AbstractTTSEngine

_REGISTRY: Dict[str, Type[AbstractTTSEngine]] = {}


def register(name: str, cls: Type[AbstractTTSEngine]) -> None:
    _REGISTRY[name] = cls


def get_engine_class(name: str) -> Type[AbstractTTSEngine]:
    if name not in _REGISTRY:
        raise ValueError(
            f'Unknown TTS engine "{name}". Available: {list(_REGISTRY.keys())}'
        )
    return _REGISTRY[name]


def available_engines() -> list[str]:
    return list(_REGISTRY.keys())


# Auto-register known engines by importing them
def _auto_register() -> None:
    for mod_name in ('omnivoice_engine', 'inflect_engine'):
        try:
            mod = importlib.import_module(f'.{mod_name}', package=__package__)
            mod.register_self()
        except ImportError:
            pass  # Optional engine not installed


_auto_register()
