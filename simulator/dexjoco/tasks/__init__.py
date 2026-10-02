__all__ = ["CONFIG_MAPPING"]

def __getattr__(name):
    if name == "CONFIG_MAPPING":
        from .mappings import CONFIG_MAPPING
        return CONFIG_MAPPING
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
