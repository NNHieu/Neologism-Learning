"""Neologism learning: parameter-efficient control of a frozen LM by training a
single new word embedding.

Implements the method from Hewitt, Tafjord, Geirhos & Kim,
"Neologism Learning for Controllability and Self-Verbalization" (arXiv:2510.08506).
"""

from .concepts import CONCEPTS, Concept
from .model_utils import (
    NeologismModel,
    load_base_model,
)

__all__ = ["CONCEPTS", "Concept", "NeologismModel", "load_base_model"]
