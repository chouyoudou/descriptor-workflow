"""Fixed periodic site-environment and frozen-cage geometry descriptors.

Only explicit fully occupied periodic coordinates and neutral element lookup values
are used. The implementation does not infer oxidation state, charge, bond order,
dopant identity, occupancy, response properties, or synthesis history.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
from math import cos, exp, log, pi
from typing import Mapping, Sequence

import numpy as np

from .elements import ElementRecord, get_element, normalize_symbol
from .poscar import PeriodicStructure

_EPS = 1.0e-12

LOCAL_CHANNELS: tuple[str, ...] = (
    "effective_coordination",
    "mean_normalized_distance",
    "radial_distortion_index",
    "offcentering",
    "orientation_anisotropy",
    "orientation_planarity",
    "orientation_linearity",
    "frozen_cage_fit_radius_ang",
    "resident_radius_residual_rel",
    "contact_strain_rms",
    "atomic_number_contrast",
    "outer_electron_contrast",
    "heteroelement_fraction",
    "local_element_effective_count",
)

STATISTICS: tuple[str, ...] = ("mean", "std", "q10", "q50", "q90")

ETA_CHANNELS: tuple[str, ...] = (
    "effective_coordination",
    "mean_normalized_distance",
    "offcentering",
    "orientation_anisotropy",
    "frozen_cage_fit_radius_ang",
    "resident_radius_residual_rel",
    "heteroelement_fraction",
)

GLOBAL_FEATURES: tuple[str, ...] = (
    "structure_valid",
    "volume_per_atom_ang3",
    "number_density_per_ang3",
    "unique_element_count",
    "covalent_radius_mean_ang",
    "covalent_radius_std_ang",
    "atomic_number_mean",
    "atomic_number_std",
    "outer_electrons_mean",
    "outer_electrons_std",
    "environment_valid_fraction",
    "environment_missing_fraction",
    "contact_weight_per_site_mean",
    "contact_weight_per_site_std",
    "heteroelement_contact_weight_fraction",
    "same_element_contact_weight_fraction",
    "multi_species_structure",
)

FEATURE_NAMES: tuple[str, ...] = (
    GLOBAL_FEATURES
    + tuple(f"{channel}_{stat}" for channel in LOCAL_CHANNELS for stat in STATISTICS)
    + tuple(f"species_eta2_{channel}" for channel in ETA_CHANNELS)
)

if len(FEATURE_NAMES) != 94 or len(set(FEATURE_NAMES)) != len(FEATURE_NAMES):
    raise RuntimeError("feature contract is not the expected 94 unique names")


class ResourceLimitError(RuntimeError):
    """Deterministic refusal when an exact configured search would exceed a guard."""


@dataclass(frozen=True)
class DescriptorConfig:
    """Numerical settings for the explicit smooth contact construction."""

    normalized_inner: float = 0.95
    normalized_outer: float = 1.35
    min_contact_weight: float = 0.05
    min_distance_ang: float = 0.10
    radius_floor_ang: float = 0.20
    max_image_range: int = 6
    max_neighbor_evaluations: int = 8_000_000

    def __post_init__(self) -> None:
        if not 0.0 < self.normalized_inner < self.normalized_outer:
            raise ValueError("require 0 < normalized_inner < normalized_outer")
        if not 0.0 < self.min_contact_weight <= 1.0:
            raise ValueError("min_contact_weight must lie in (0, 1]")
        if self.min_distance_ang <= 0.0:
            raise ValueError("min_distance_ang must be positive")
        if self.radius_floor_ang <= 0.0:
            raise ValueError("radius_floor_ang must be positive")
        if self.max_image_range < 1:
            raise ValueError("max_image_range must be positive")
        if self.max_neighbor_evaluations < 1:
            raise ValueError("max_neighbor_evaluations must be positive")


@dataclass(frozen=True)
class SiteEnvironment:
    """Per-site channels and typed applicability information."""

    site_index: int
    symbol: str
    valid: bool
    missing_reason: str | None
    contact_weight: float
    channels: Mapping[str, float]

    def __post_init__(self) -> None:
        if tuple(self.channels) != LOCAL_CHANNELS:
            raise ValueError("site channel order differs from contract")
        for value in self.channels.values():
            if self.valid and not np.isfinite(float(value)):
                raise ValueError("valid site environment contains a non-finite channel")


@dataclass(frozen=True)
class DescriptorResult:
    """Fixed structure-level descriptor with JSON-null typed missing values."""

    feature_names: tuple[str, ...]
    values: tuple[float | None, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("feature_names differ from the fixed contract")
        if len(self.values) != len(self.feature_names):
            raise ValueError("feature_names and values differ in length")
        for value in self.values:
            if value is not None and not np.isfinite(float(value)):
                raise ValueError("descriptor contains a non-finite non-null value")

    def as_dict(self) -> dict[str, float | None]:
        return dict(zip(self.feature_names, self.values))

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema": "periodic-site-environment/1",
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "features": self.as_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProbeEvaluation:
    """Variable-length explicit probe output, separate from the fixed descriptor."""

    probes: Mapping[str, Mapping[str, float | int | None]]
    metadata: Mapping[str, object]

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema": "explicit-frozen-cage-probes/1",
            "probes": {key: dict(value) for key, value in self.probes.items()},
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _Contact:
    neighbor_index: int
    distance_ang: float
    normalized_distance: float
    weight: float
    unit_direction: np.ndarray


@dataclass(frozen=True)
class _Analysis:
    environments: tuple[SiteEnvironment, ...]
    records: tuple[ElementRecord, ...]
    image_ranges: tuple[int, int, int]
    neighbor_evaluations: int
    heteroelement_weight: float
    total_contact_weight: float


def smooth_contact_weight(normalized_distance: float, config: DescriptorConfig) -> float:
    """Compact cosine contact weight; not a bond order or interaction energy."""

    s = float(normalized_distance)
    if s <= config.normalized_inner:
        return 1.0
    if s >= config.normalized_outer:
        return 0.0
    phase = (s - config.normalized_inner) / (
        config.normalized_outer - config.normalized_inner
    )
    return 0.5 * (1.0 + cos(pi * phase))


def compute_site_environments(
    structure: PeriodicStructure,
    config: DescriptorConfig | None = None,
) -> tuple[SiteEnvironment, ...]:
    """Return ordered explicit-site environments for diagnostics."""

    return _analyze_structure(structure, config or DescriptorConfig()).environments


def featurize_structure(
    structure: PeriodicStructure,
    config: DescriptorConfig | None = None,
) -> DescriptorResult:
    """Compute the fixed 94-scalar descriptor contract."""

    cfg = config or DescriptorConfig()
    analysis = _analyze_structure(structure, cfg)
    envs = analysis.environments
    records = analysis.records
    n = structure.n_sites

    radii = np.asarray([record.covalent_radius_angstrom for record in records], dtype=float)
    atomic_numbers = np.asarray([record.atomic_number for record in records], dtype=float)
    outer = np.asarray([record.outer_electrons for record in records], dtype=float)
    contact_weights = np.asarray([env.contact_weight for env in envs], dtype=float)
    valid_count = sum(env.valid for env in envs)

    values: dict[str, float | None] = {
        "structure_valid": 1.0,
        "volume_per_atom_ang3": structure.volume / n,
        "number_density_per_ang3": n / structure.volume,
        "unique_element_count": float(len(set(structure.species))),
        "covalent_radius_mean_ang": float(np.mean(radii)),
        "covalent_radius_std_ang": float(np.std(radii)),
        "atomic_number_mean": float(np.mean(atomic_numbers)),
        "atomic_number_std": float(np.std(atomic_numbers)),
        "outer_electrons_mean": float(np.mean(outer)),
        "outer_electrons_std": float(np.std(outer)),
        "environment_valid_fraction": valid_count / n,
        "environment_missing_fraction": 1.0 - valid_count / n,
        "contact_weight_per_site_mean": float(np.mean(contact_weights)),
        "contact_weight_per_site_std": float(np.std(contact_weights)),
        "heteroelement_contact_weight_fraction": (
            analysis.heteroelement_weight / analysis.total_contact_weight
            if analysis.total_contact_weight > _EPS else None
        ),
        "same_element_contact_weight_fraction": (
            1.0 - analysis.heteroelement_weight / analysis.total_contact_weight
            if analysis.total_contact_weight > _EPS else None
        ),
        "multi_species_structure": float(len(set(structure.species)) > 1),
    }

    for channel in LOCAL_CHANNELS:
        channel_values = [
            float(env.channels[channel])
            for env in envs
            if env.valid and np.isfinite(float(env.channels[channel]))
        ]
        stats = _summary_statistics(channel_values)
        for stat, number in zip(STATISTICS, stats):
            values[f"{channel}_{stat}"] = number

    for channel in ETA_CHANNELS:
        values[f"species_eta2_{channel}"] = _species_eta_squared(
            structure.species, envs, channel
        )

    ordered_values = tuple(values[name] for name in FEATURE_NAMES)
    missing_reasons = Counter(
        env.missing_reason for env in envs
        if not env.valid and env.missing_reason is not None
    )
    metadata = {
        "descriptor": "periodic-site-environment",
        "version": 1,
        "paper_inspiration": "10.1039/d2ra06962h",
        "n_sites": n,
        "n_valid_environments": valid_count,
        "n_missing_environments": n - valid_count,
        "missing_reasons": dict(sorted(missing_reasons.items())),
        "image_ranges": list(analysis.image_ranges),
        "neighbor_evaluations": analysis.neighbor_evaluations,
        "defined_feature_count": sum(value is not None for value in ordered_values),
        "feature_count": len(FEATURE_NAMES),
        "config": asdict(cfg),
        "scope_note": (
            "Static fully occupied periodic geometry plus neutral element lookup only. "
            "No oxidation state, dopant occupancy/site, relaxed substitution energy, "
            "optical/thermal response, or synthesis history is inferred."
        ),
    }
    return DescriptorResult(FEATURE_NAMES, ordered_values, metadata)


def evaluate_probe_elements(
    structure: PeriodicStructure,
    probe_symbols: Sequence[str],
    config: DescriptorConfig | None = None,
) -> ProbeEvaluation:
    """Evaluate caller-named neutral radii in the unchanged explicit cages."""

    cfg = config or DescriptorConfig()
    analysis = _analyze_structure(structure, cfg)
    valid_envs = [env for env in analysis.environments if env.valid]
    probes: dict[str, dict[str, float | int | None]] = {}
    seen: set[str] = set()
    for token in probe_symbols:
        symbol = normalize_symbol(token)
        if symbol in seen:
            continue
        seen.add(symbol)
        record = get_element(symbol)
        probe_residuals: list[float] = []
        resident_residuals: list[float] = []
        for env in valid_envs:
            fit_radius = float(env.channels["frozen_cage_fit_radius_ang"])
            probe_residual = (
                record.covalent_radius_angstrom - fit_radius
            ) / max(record.covalent_radius_angstrom, cfg.radius_floor_ang)
            probe_residuals.append(probe_residual)
            resident_residuals.append(float(env.channels["resident_radius_residual_rel"]))
        if probe_residuals:
            probe_arr = np.asarray(probe_residuals, dtype=float)
            resident_arr = np.asarray(resident_residuals, dtype=float)
            probes[symbol] = {
                "probe_atomic_number": record.atomic_number,
                "probe_covalent_radius_ang": record.covalent_radius_angstrom,
                "valid_cage_count": len(valid_envs),
                "coverage_fraction": len(valid_envs) / structure.n_sites,
                "signed_residual_mean": float(np.mean(probe_arr)),
                "absolute_residual_mean": float(np.mean(np.abs(probe_arr))),
                "absolute_residual_q90": float(
                    np.quantile(np.abs(probe_arr), 0.90, method="inverted_cdf")
                ),
                "absolute_residual_delta_vs_resident_mean": float(
                    np.mean(np.abs(probe_arr) - np.abs(resident_arr))
                ),
            }
        else:
            probes[symbol] = {
                "probe_atomic_number": record.atomic_number,
                "probe_covalent_radius_ang": record.covalent_radius_angstrom,
                "valid_cage_count": 0,
                "coverage_fraction": 0.0,
                "signed_residual_mean": None,
                "absolute_residual_mean": None,
                "absolute_residual_q90": None,
                "absolute_residual_delta_vs_resident_mean": None,
            }
    return ProbeEvaluation(
        probes=probes,
        metadata={
            "explicit_probe_count": len(probes),
            "n_sites": structure.n_sites,
            "image_ranges": list(analysis.image_ranges),
            "neighbor_evaluations": analysis.neighbor_evaluations,
            "config": asdict(cfg),
            "scope_note": (
                "Caller-supplied neutral-radius fit in frozen cages only; no occupancy, "
                "site preference, relaxation, energy, valence, or synthesizability claim."
            ),
        },
    )


def _analyze_structure(structure: PeriodicStructure, cfg: DescriptorConfig) -> _Analysis:
    records = tuple(get_element(symbol) for symbol in structure.species)
    contacts, image_ranges, evaluations = _enumerate_contacts(structure, records, cfg)
    environments: list[SiteEnvironment] = []
    total_weight = 0.0
    hetero_weight = 0.0
    for i, site_contacts in enumerate(contacts):
        env = _environment_from_contacts(i, records, site_contacts, cfg)
        environments.append(env)
        total_weight += env.contact_weight
        hetero_weight += sum(
            contact.weight for contact in site_contacts
            if records[contact.neighbor_index].symbol != records[i].symbol
        )
    return _Analysis(
        environments=tuple(environments),
        records=records,
        image_ranges=image_ranges,
        neighbor_evaluations=evaluations,
        heteroelement_weight=float(hetero_weight),
        total_contact_weight=float(total_weight),
    )


def _enumerate_contacts(
    structure: PeriodicStructure,
    records: Sequence[ElementRecord],
    cfg: DescriptorConfig,
) -> tuple[list[list[_Contact]], tuple[int, int, int], int]:
    n = structure.n_sites
    radii = np.asarray([record.covalent_radius_angstrom for record in records], dtype=float)
    pair_radii = radii[:, None] + radii[None, :]
    max_cutoff = cfg.normalized_outer * float(np.max(pair_radii))

    inv_lattice = np.linalg.inv(structure.lattice)
    plane_spacings = np.asarray(
        [1.0 / max(np.linalg.norm(inv_lattice[:, axis]), _EPS) for axis in range(3)],
        dtype=float,
    )
    ranges = np.maximum(1, np.floor(max_cutoff / plane_spacings).astype(int) + 1)
    if np.any(ranges > cfg.max_image_range):
        raise ResourceLimitError(
            "required_periodic_image_range_exceeds_limit:"
            + ",".join(str(int(x)) for x in ranges)
        )
    shifts = tuple(
        np.asarray(shift, dtype=int)
        for shift in product(*(range(-int(span), int(span) + 1) for span in ranges))
    )
    evaluations = len(shifts) * n * n
    if evaluations > cfg.max_neighbor_evaluations:
        raise ResourceLimitError(
            f"neighbor_evaluations_exceed_limit:{evaluations}>{cfg.max_neighbor_evaluations}"
        )

    contacts: list[list[_Contact]] = [[] for _ in range(n)]
    frac = structure.frac_coords
    zero_shift = np.zeros(3, dtype=int)
    for shift in shifts:
        diff = frac[None, :, :] + shift[None, None, :] - frac[:, None, :]
        cart = diff @ structure.lattice
        distances = np.linalg.norm(cart, axis=2)
        excluded = np.zeros((n, n), dtype=bool)
        if np.array_equal(shift, zero_shift):
            np.fill_diagonal(excluded, True)
        overlap = (distances < cfg.min_distance_ang) & ~excluded
        if np.any(overlap):
            center, neighbor = np.argwhere(overlap)[0]
            raise ValueError(
                f"overlapping_periodic_sites:center={int(center)},neighbor={int(neighbor)},"
                f"shift={tuple(int(x) for x in shift)}"
            )
        normalized = distances / pair_radii
        candidate = (~excluded) & (normalized < cfg.normalized_outer)
        for center in range(n):
            for neighbor in np.flatnonzero(candidate[center]):
                distance = float(distances[center, neighbor])
                s = float(normalized[center, neighbor])
                weight = smooth_contact_weight(s, cfg)
                if weight <= 0.0:
                    continue
                contacts[center].append(
                    _Contact(
                        neighbor_index=int(neighbor),
                        distance_ang=distance,
                        normalized_distance=s,
                        weight=weight,
                        unit_direction=cart[center, neighbor] / distance,
                    )
                )
    return contacts, tuple(int(x) for x in ranges), evaluations


def _invalid_channels() -> dict[str, float]:
    return {name: float("nan") for name in LOCAL_CHANNELS}


def _environment_from_contacts(
    site_index: int,
    records: Sequence[ElementRecord],
    contacts: Sequence[_Contact],
    cfg: DescriptorConfig,
) -> SiteEnvironment:
    total_weight = float(sum(contact.weight for contact in contacts))
    center = records[site_index]
    if total_weight < cfg.min_contact_weight:
        return SiteEnvironment(
            site_index=site_index,
            symbol=center.symbol,
            valid=False,
            missing_reason="no_weighted_contacts",
            contact_weight=total_weight,
            channels=_invalid_channels(),
        )

    p = np.asarray([contact.weight for contact in contacts], dtype=float) / total_weight
    distances = np.asarray([contact.distance_ang for contact in contacts], dtype=float)
    normalized = np.asarray([contact.normalized_distance for contact in contacts], dtype=float)
    directions = np.asarray([contact.unit_direction for contact in contacts], dtype=float)
    neighbor_records = [records[contact.neighbor_index] for contact in contacts]
    neighbor_radii = np.asarray(
        [record.covalent_radius_angstrom for record in neighbor_records], dtype=float
    )
    neighbor_z = np.asarray([record.atomic_number for record in neighbor_records], dtype=float)
    neighbor_outer = np.asarray([record.outer_electrons for record in neighbor_records], dtype=float)

    mean_distance = float(np.dot(p, distances))
    radial_distortion = float(
        np.dot(p, np.abs(distances - mean_distance)) / max(mean_distance, _EPS)
    )
    first_moment = np.sum(p[:, None] * directions, axis=0)
    offcentering = float(np.linalg.norm(first_moment))
    orientation = np.einsum("i,ij,ik->jk", p, directions, directions)
    eigenvalues = np.linalg.eigvalsh(orientation)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    anisotropy = float(1.5 * np.sum((eigenvalues - 1.0 / 3.0) ** 2))
    planarity = float(eigenvalues[1] - eigenvalues[2])
    linearity = float(eigenvalues[0] - eigenvalues[1])
    fit_radius = float(np.dot(p, distances - neighbor_radii))
    radius_residual = (
        center.covalent_radius_angstrom - fit_radius
    ) / max(center.covalent_radius_angstrom, cfg.radius_floor_ang)
    contact_strain = float(np.sqrt(np.dot(p, (normalized - 1.0) ** 2)))
    z_contrast = float((center.atomic_number - np.dot(p, neighbor_z)) / 118.0)
    outer_contrast = float((center.outer_electrons - np.dot(p, neighbor_outer)) / 15.0)
    hetero = np.asarray(
        [record.symbol != center.symbol for record in neighbor_records], dtype=float
    )
    hetero_fraction = float(np.dot(p, hetero))

    by_species: dict[str, float] = {}
    for probability, record in zip(p, neighbor_records):
        by_species[record.symbol] = by_species.get(record.symbol, 0.0) + float(probability)
    entropy = -sum(q * log(q) for q in by_species.values() if q > 0.0)
    effective_elements = float(exp(entropy))

    channels = {
        "effective_coordination": total_weight,
        "mean_normalized_distance": float(np.dot(p, normalized)),
        "radial_distortion_index": radial_distortion,
        "offcentering": offcentering,
        "orientation_anisotropy": anisotropy,
        "orientation_planarity": planarity,
        "orientation_linearity": linearity,
        "frozen_cage_fit_radius_ang": fit_radius,
        "resident_radius_residual_rel": float(radius_residual),
        "contact_strain_rms": contact_strain,
        "atomic_number_contrast": z_contrast,
        "outer_electron_contrast": outer_contrast,
        "heteroelement_fraction": hetero_fraction,
        "local_element_effective_count": effective_elements,
    }
    return SiteEnvironment(
        site_index=site_index,
        symbol=center.symbol,
        valid=True,
        missing_reason=None,
        contact_weight=total_weight,
        channels=channels,
    )


def _summary_statistics(
    values: Sequence[float],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (None, None, None, None, None)
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.quantile(arr, 0.10, method="inverted_cdf")),
        float(np.quantile(arr, 0.50, method="inverted_cdf")),
        float(np.quantile(arr, 0.90, method="inverted_cdf")),
    )


def _species_eta_squared(
    species: Sequence[str],
    environments: Sequence[SiteEnvironment],
    channel: str,
) -> float | None:
    selected = [
        (symbol, float(env.channels[channel]))
        for symbol, env in zip(species, environments)
        if env.valid and np.isfinite(float(env.channels[channel]))
    ]
    if not selected or len({symbol for symbol, _ in selected}) < 2:
        return None
    values = np.asarray([value for _, value in selected], dtype=float)
    overall = float(np.mean(values))
    total = float(np.sum((values - overall) ** 2))
    if total <= _EPS:
        return 0.0
    between = 0.0
    for symbol in sorted({symbol for symbol, _ in selected}):
        group = np.asarray(
            [value for current, value in selected if current == symbol], dtype=float
        )
        between += len(group) * (float(np.mean(group)) - overall) ** 2
    return float(np.clip(between / total, 0.0, 1.0))
