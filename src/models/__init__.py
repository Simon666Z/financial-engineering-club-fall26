"""Model modules, with optional backends imported only when requested."""

__all__ = ["ShowcaseBundle", "train_showcase"]


def __getattr__(name):
    if name in __all__:
        from .showcase import ShowcaseBundle, train_showcase
        exports = {"ShowcaseBundle": ShowcaseBundle, "train_showcase": train_showcase}
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
