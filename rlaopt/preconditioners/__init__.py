import importlib

modules_to_import = [
    "rlaopt.preconditioners.configs",
    "rlaopt.preconditioners.preconditioner",
    "rlaopt.preconditioners.preconditioner_factory",
]

__all__ = []
for module in modules_to_import:
    mod = importlib.import_module(module)
    components = getattr(mod, "__all__", [])
    for component in components:
        globals()[component] = getattr(mod, component)
        __all__.append(component)
