"""Interpretable molecular-crystal packing scalars for ordinary POSCAR files."""

from .analysis import DESCRIPTOR_NAMES, DESCRIPTOR_UNITS, AnalysisResult, analyze_structure
from .api import (
    DESCRIPTOR_FUNCTIONS,
    SCHEMA,
    TASK_ID,
    VERSION,
    component_shape_anisotropy,
    compute_all,
    compute_descriptor,
    contact_pair_specificity,
    coordination_shell_orientation_anisotropy,
    coordination_shell_radial_distortion,
    heteroelement_contact_fraction,
    interfragment_contact_anisotropy,
    interfragment_contact_compactness,
    interfragment_contact_coordination,
    spacer_atom_fraction,
    spacer_path_extension,
)
from .model import DescriptorConfig, DescriptorError, Structure
from .poscar import parse_poscar_text, read_poscar

__all__ = [
    "AnalysisResult",
    "DESCRIPTOR_FUNCTIONS",
    "DESCRIPTOR_NAMES",
    "DESCRIPTOR_UNITS",
    "DescriptorConfig",
    "DescriptorError",
    "SCHEMA",
    "Structure",
    "TASK_ID",
    "VERSION",
    "analyze_structure",
    "component_shape_anisotropy",
    "compute_all",
    "compute_descriptor",
    "contact_pair_specificity",
    "coordination_shell_orientation_anisotropy",
    "coordination_shell_radial_distortion",
    "heteroelement_contact_fraction",
    "interfragment_contact_anisotropy",
    "interfragment_contact_compactness",
    "interfragment_contact_coordination",
    "parse_poscar_text",
    "read_poscar",
    "spacer_atom_fraction",
    "spacer_path_extension",
]
