"""Top-level package for comfy_archi."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Kun"""
__email__ = "kun_g@msn.com"
__version__ = "0.0.1"

from .src.comfy_archi.nodes import NODE_CLASS_MAPPINGS
from .src.comfy_archi.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
