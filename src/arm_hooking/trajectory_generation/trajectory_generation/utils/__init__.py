import pkgutil
import importlib

__all__ = []

# Automatically import all modules in this package
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    # Import the module
    module = importlib.import_module(f"{__name__}.{module_name}")
    # Add all public symbols (those not starting with "_") to this namespace
    for attr in dir(module):
        if not attr.startswith("_"):
            globals()[attr] = getattr(module, attr)
            __all__.append(attr)

