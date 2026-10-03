import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336-training")
except importlib.metadata.PackageNotFoundError:
    pass
