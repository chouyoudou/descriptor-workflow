"""Radius-normalized non-bonded interfragment contact statistics."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .model import BondEdge, Component, Contact, DescriptorConfig, Structure
from .periodic import (
    _add_shift,
    _is_zero_shift,
    _iter_candidate_pairs,
    _neg_shift,
    _radii,
    _sub_shift,
)

def build_contacts(
    structure: Structure,
    config: DescriptorConfig,
    bond_edges: list[BondEdge],
    components: list[Component],
    component_of: list[int],
) -> list[Contact]:
    _, _, vdw = _radii(structure)
    max_cutoff = float(2.0 * config.contact_ratio_max * np.max(vdw))
    bond_keys = {(edge.i, edge.j, edge.shift) for edge in bond_edges}
    contacts: list[Contact] = []
    for i, j, shift, vector, distance in _iter_candidate_pairs(structure, max_cutoff, config):
        if (i, j, shift) in bond_keys:
            continue
        radius_sum = float(vdw[i] + vdw[j])
        ratio = distance / radius_sum
        if ratio < config.contact_ratio_min - 1e-12 or ratio > config.contact_ratio_max + 1e-12:
            continue
        ci = component_of[i]
        cj = component_of[j]
        component_i = components[ci]
        component_j = components[cj]
        oi = component_i.offsets[i]
        oj = component_j.offsets[j]
        component_shift = _add_shift(_sub_shift(shift, oj), oi)
        if ci == cj:
            if component_i.periodic:
                continue
            if _is_zero_shift(component_shift):
                continue
        closeness = (config.contact_ratio_max - ratio) / (
            config.contact_ratio_max - config.contact_ratio_full
        )
        closeness = float(np.clip(closeness, 0.0, 1.0))
        if closeness <= 1e-15:
            continue
        contacts.append(
            Contact(
                i=i,
                j=j,
                shift=shift,
                vector=vector.copy(),
                distance=distance,
                normalized_vdw_distance=ratio,
                closeness=closeness,
                component_i=ci,
                component_j=cj,
                component_shift=component_shift,
            )
        )
    return contacts


def contact_metrics(
    structure: Structure,
    components: list[Component],
    contacts: list[Contact],
) -> tuple[float, float, float, float, float]:
    """Return compactness, coordination, anisotropy, hetero fraction, specificity."""

    if not contacts:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    closeness = np.asarray([contact.closeness for contact in contacts], dtype=float)
    total = float(np.sum(closeness))
    compactness = float(np.mean(closeness))

    directions = np.asarray([contact.vector / contact.distance for contact in contacts], dtype=float)
    tensor = np.einsum("n,ni,nj->ij", closeness, directions, directions) / total
    eig = np.linalg.eigvalsh(tensor)
    anisotropy = float(np.clip(1.5 * np.sum((eig - 1.0 / 3.0) ** 2), 0.0, 1.0))

    hetero = sum(
        contact.closeness
        for contact in contacts
        if structure.species[contact.i] != structure.species[contact.j]
    ) / total

    pair_weights: dict[tuple[str, str], float] = defaultdict(float)
    for contact in contacts:
        key = tuple(sorted((structure.species[contact.i], structure.species[contact.j])))
        pair_weights[key] += contact.closeness
    if len(pair_weights) == 1:
        specificity = 1.0
    else:
        probabilities = np.asarray(list(pair_weights.values()), dtype=float) / total
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        specificity = 1.0 - entropy / math.log(len(pair_weights))
        specificity = float(np.clip(specificity, 0.0, 1.0))

    neighbor_weights: list[dict[tuple[int, tuple[int, int, int]], float]] = [
        defaultdict(float) for _ in components
    ]
    for contact in contacts:
        t = contact.component_shift
        neighbor_weights[contact.component_i][(contact.component_j, t)] += contact.closeness
        neighbor_weights[contact.component_j][(contact.component_i, _neg_shift(t))] += contact.closeness
    effective_numbers = []
    for weights in neighbor_weights:
        values = np.asarray(list(weights.values()), dtype=float)
        if values.size:
            effective_numbers.append(float(np.sum(values) ** 2 / np.sum(values**2)))
    coordination = float(np.mean(effective_numbers)) if effective_numbers else 0.0
    return compactness, coordination, anisotropy, float(np.clip(hetero, 0.0, 1.0)), specificity

