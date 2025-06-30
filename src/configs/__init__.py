# configs/__init__.py
from .configs import Configs


def bool_(value):
    return True if value == "True" or value == "true" else False


__all__ = ["Configs", "bool_"]
