"""Dataset acquisition and preparation entry points."""

from .uci import fetch_uci, prepare_uci
from .unga import prepare_unga, register_unga_download

__all__ = [
    "fetch_uci",
    "prepare_uci",
    "prepare_unga",
    "register_unga_download",
]

