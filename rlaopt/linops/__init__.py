import importlib

modules_to_import = [
    ".distributed",
    ".simple",
    ".types",
]

__all__ = []
for module in modules_to_import:
    mod = importlib.import_module(module, package=__package__)
    components = getattr(mod, "__all__", [])
    for component in components:
        globals()[component] = getattr(mod, component)
        __all__.append(component)
