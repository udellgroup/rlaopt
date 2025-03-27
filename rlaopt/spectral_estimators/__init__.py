"Spectral estimators module __init__.py file."

import importlib

modules_to_import = [
    "rlaopt.spectral_estimators.frobenius_norm",
    "rlaopt.spectral_estimators.spectral_norm",
    "rlaopt.spectral_estimators.trace",
]

__all__ = []
for module in modules_to_import:
    mod = importlib.import_module(module)
    components = getattr(mod, "__all__", [])
    for component in components:
        globals()[component] = getattr(mod, component)
        __all__.append(component)
