"""MCIE experimental pipeline.

The public notation is fixed throughout the package:

* AR: ``D <= beta``;
* NR: ``beta < D < alpha``;
* CR: ``D >= alpha``.
"""

from .model import PairInstance, Region, SituationTable, classify_degree

__all__ = [
    "PairInstance",
    "Region",
    "SituationTable",
    "classify_degree",
]

__version__ = "0.1.0"

