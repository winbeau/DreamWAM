"""Public model exports, loaded only when a model is requested.

Pure configuration and experiment planning must work without importing torch.
"""

from importlib import import_module

__all__ = ["DreamWAMConfig", "DreamWAMJoint", "DreamWAMUncond"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(".model", __name__), name)
    globals()[name] = value
    return value
