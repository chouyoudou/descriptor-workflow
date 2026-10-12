"""Cheap periodic element-reference contact contrast descriptors.

Fixed scalar (ERCCD): for every unique periodic unlike-element near contact,

    delta_Z = |Z_i-Z_j|/(Z_i+Z_j)
    delta_r = |r_i-r_j|/(r_i+r_j)
    c_ij = sqrt((delta_Z**2 + delta_r**2)/2)
    ERCCD = sum(c_ij)/N_atoms

A contact obeys d_ij <= 1.25*(r_i+r_j) by default. The meaningful extension
is M = sum(c_ij*u_ij outer u_ij)/N_atoms; trace(M) equals ERCCD. Inputs are
static periodic coordinates plus independent neutral-element references only.
No optical, device, film-thickness, resonator, illumination, charge, spin,
energy, or process response is inferred.
"""

from .erccd_core import analyze_poscar, analyze_structure
from .erccd_elements import (
    ATOMIC_NUMBERS,
    CHEMICAL_SYMBOLS,
    COVALENT_RADII_ANGSTROM,
    DescriptorError,
    canonical_symbol,
    covalent_radius_angstrom,
)
from .erccd_qualification import run_real_structure_qualification
from .erccd_structure import (
    DescriptorConfig,
    Structure,
    parse_poscar,
    structure_from_jarvis_record,
)

__all__ = [
    "ATOMIC_NUMBERS",
    "CHEMICAL_SYMBOLS",
    "COVALENT_RADII_ANGSTROM",
    "DescriptorConfig",
    "DescriptorError",
    "Structure",
    "analyze_poscar",
    "analyze_structure",
    "canonical_symbol",
    "covalent_radius_angstrom",
    "parse_poscar",
    "run_real_structure_qualification",
    "structure_from_jarvis_record",
]
