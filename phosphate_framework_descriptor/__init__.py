"""Periodic framework/polyhedron/void descriptors inspired by rare-earth phosphates."""

from .descriptor import DescriptorConfig, DescriptorResult, featurize_structure
from .poscar import PeriodicStructure, parse_poscar_text, read_poscar, write_poscar

__all__ = [
    "DescriptorConfig",
    "DescriptorResult",
    "PeriodicStructure",
    "featurize_structure",
    "parse_poscar_text",
    "read_poscar",
    "write_poscar",
]
__version__ = "0.1.0"
