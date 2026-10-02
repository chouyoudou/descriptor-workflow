"""Public orchestration for static molecular-crystal packing scalars.

The implementation is clean-room and uses only NumPy plus the local radius
table. It does not assign charge, oxidation state, bond order, magnetic state,
or any measured-property label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contacts import build_contacts, contact_metrics
from .coordination import coordination_shell_metrics
from .model import DescriptorConfig, DescriptorError, Structure
from .periodic import build_bonds, find_components
from .spacer import component_shape_anisotropy, spacer_metrics

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "spacer_path_extension",
    "spacer_atom_fraction",
    "component_shape_anisotropy",
    "coordination_shell_radial_distortion",
    "coordination_shell_orientation_anisotropy",
    "interfragment_contact_compactness",
    "interfragment_contact_coordination",
    "interfragment_contact_anisotropy",
    "heteroelement_contact_fraction",
    "contact_pair_specificity",
)

DESCRIPTOR_UNITS: dict[str, str] = {name: "dimensionless" for name in DESCRIPTOR_NAMES}

@dataclass(frozen=True)
class AnalysisResult:
    values: dict[str, float]
    n_bonds: int
    n_components: int
    n_periodic_components: int
    n_contacts: int


def analyze_structure(
    structure: Structure,
    config: DescriptorConfig | None = None,
) -> AnalysisResult:
    config = config or DescriptorConfig()
    config.validate()
    if structure.n_atoms < 1:
        raise DescriptorError("structure has no atoms")
    if structure.volume <= 1e-8:
        raise DescriptorError("structure volume is not positive")

    bond_edges, adjacency = build_bonds(structure, config)
    components, component_of = find_components(structure, adjacency)
    spacer_extension, spacer_fraction = spacer_metrics(structure, components, adjacency, config)
    shape = component_shape_anisotropy(structure, components)
    radial, orientation = coordination_shell_metrics(structure, adjacency)
    contacts = build_contacts(structure, config, bond_edges, components, component_of)
    compactness, contact_coordination, contact_anisotropy, hetero, specificity = contact_metrics(
        structure, components, contacts
    )
    values = {
        "spacer_path_extension": spacer_extension,
        "spacer_atom_fraction": spacer_fraction,
        "component_shape_anisotropy": shape,
        "coordination_shell_radial_distortion": radial,
        "coordination_shell_orientation_anisotropy": orientation,
        "interfragment_contact_compactness": compactness,
        "interfragment_contact_coordination": contact_coordination,
        "interfragment_contact_anisotropy": contact_anisotropy,
        "heteroelement_contact_fraction": hetero,
        "contact_pair_specificity": specificity,
    }
    for name, value in values.items():
        if not isinstance(value, float) or not math.isfinite(value):
            raise DescriptorError(f"descriptor {name} is not a finite Python float")
    return AnalysisResult(
        values=values,
        n_bonds=len(bond_edges),
        n_components=len(components),
        n_periodic_components=sum(component.periodic for component in components),
        n_contacts=len(contacts),
    )
