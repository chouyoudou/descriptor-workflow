"""Trans-pair decomposition for a declared six-vector shell."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

try:
    from .otpd_types import DescriptorError, Settings, SiteResult, canonical_symbol, radius_table
    from .otpd_geometry import (
        _angle_deg, _canonical_axis, _correlation, _cosine_similarity,
        _pair_metric, _perfect_matchings, _unit,
    )
except ImportError:
    from otpd_types import DescriptorError, Settings, SiteResult, canonical_symbol, radius_table
    from otpd_geometry import (
        _angle_deg, _canonical_axis, _correlation, _cosine_similarity,
        _pair_metric, _perfect_matchings, _unit,
    )

def analyze_six_vectors(
    center_symbol: str,
    neighbor_symbols: Sequence[str],
    vectors: Sequence[Sequence[float] | np.ndarray],
    *,
    radii: Mapping[str, float] | None = None,
    settings: Settings | None = None,
    center_index: int | None = None,
    neighbor_keys: Sequence[tuple[int, int, int, int]] | None = None,
    donor_plane_normals: Sequence[np.ndarray | None] | None = None,
) -> SiteResult | None:
    settings = settings or Settings()
    settings.validate()
    radius_map = radius_table(radii)
    center_symbol = canonical_symbol(center_symbol)
    symbols = [canonical_symbol(s) for s in neighbor_symbols]
    if len(symbols) != 6 or len(vectors) != 6:
        raise DescriptorError("exactly six donor vectors are required")
    vec = [np.asarray(v, dtype=float) for v in vectors]
    lengths = [float(np.linalg.norm(v)) for v in vec]
    if any((not math.isfinite(d) or d <= settings.numerical_tolerance) for d in lengths):
        raise DescriptorError("donor vectors must be finite and nonzero")
    units = [v / d for v, d in zip(vec, lengths, strict=True)]
    if neighbor_keys is None:
        keys = [(i, 0, 0, 0) for i in range(6)]
    else:
        if len(neighbor_keys) != 6:
            raise DescriptorError("neighbor_keys length mismatch")
        keys = list(neighbor_keys)

    best: tuple[tuple[float, float, tuple[tuple[int, int], ...]], tuple[tuple[int, int], ...], list[float]] | None = None
    for matching in _perfect_matchings(tuple(range(6))):
        bends = [180.0 - _angle_deg(units[i], units[j]) for i, j in matching]
        objective = (sum(b * b for b in bends), max(bends), matching)
        if best is None or objective < best[0]:
            best = (objective, matching, bends)
    assert best is not None
    _, matching, trans_bends = best
    trans_rms = math.sqrt(sum(b * b for b in trans_bends) / 3.0)
    paired = {tuple(sorted(pair)) for pair in matching}
    cis_angles = [
        _angle_deg(units[i], units[j])
        for i in range(6)
        for j in range(i + 1, 6)
        if (i, j) not in paired
    ]
    cis_deviations = [a - 90.0 for a in cis_angles]
    cis_rms = math.sqrt(sum(x * x for x in cis_deviations) / len(cis_deviations))
    if trans_rms > settings.trans_rms_max_deg or cis_rms > settings.cis_rms_max_deg:
        return None

    pair_means = [(lengths[i] + lengths[j]) / 2.0 for i, j in matching]
    pair_asymmetries = [abs(lengths[i] - lengths[j]) / pair_means[k] for k, (i, j) in enumerate(matching)]
    pair_asymmetry_rms = math.sqrt(sum(x * x for x in pair_asymmetries) / 3.0)
    pair_metrics = _pair_metric(pair_means)

    center_radius = radius_map[center_symbol]
    normalized_lengths = [
        lengths[i] / (center_radius + radius_map[symbols[i]]) for i in range(6)
    ]
    normalized_pair_means = [
        (normalized_lengths[i] + normalized_lengths[j]) / 2.0 for i, j in matching
    ]
    normalized_metrics = _pair_metric(normalized_pair_means)

    mean_length = sum(lengths) / 6.0
    baur = sum(abs(d - mean_length) for d in lengths) / (6.0 * mean_length)
    bond_angle_variance = sum(x * x for x in cis_deviations) / (len(cis_deviations) - 1)
    strain_bend_correlation = _correlation(pair_means, trans_bends)

    radius_deltas: list[float] = []
    bond_deltas: list[float] = []
    donor_mismatch_pairs = 0
    for i, j in matching:
        ri, rj = radius_map[symbols[i]], radius_map[symbols[j]]
        di, dj = lengths[i], lengths[j]
        if rj < ri:
            ri, rj = rj, ri
            di, dj = dj, di
        dr = (rj - ri) / ((ri + rj) / 2.0)
        if abs(dr) <= settings.numerical_tolerance:
            continue
        dd = (dj - di) / ((di + dj) / 2.0)
        radius_deltas.append(dr)
        bond_deltas.append(dd)
        donor_mismatch_pairs += 1
    donor_size_bond_correlation = _cosine_similarity(radius_deltas, bond_deltas)

    plane_twists: list[float] = []
    if donor_plane_normals is not None:
        if len(donor_plane_normals) != 6:
            raise DescriptorError("donor_plane_normals length mismatch")
        for i, j in matching:
            ni = donor_plane_normals[i]
            nj = donor_plane_normals[j]
            if ni is None or nj is None:
                continue
            cosine = abs(float(np.dot(_unit(ni), _unit(nj))))
            plane_twists.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    plane_twist_mean = None if not plane_twists else float(sum(plane_twists) / len(plane_twists))

    sorted_pair_indices = sorted(
        range(3),
        key=lambda k: (
            round(pair_means[k], 12),
            tuple(sorted((keys[matching[k][0]], keys[matching[k][1]]))),
        ),
    )
    short_pair_k = sorted_pair_indices[0]
    short_i, short_j = matching[short_pair_k]
    short_axis = _canonical_axis(units[short_i] - units[short_j])

    values: dict[str, float | None] = {
        "pair_range": pair_metrics["pair_range"],
        "uniaxial_polarity": pair_metrics["uniaxial_polarity"],
        "short_pair_log_compression": pair_metrics["short_pair_log_compression"],
        "long_pair_log_elongation": pair_metrics["long_pair_log_elongation"],
        "pair_asymmetry_rms": pair_asymmetry_rms,
        "trans_bend_rms_deg": trans_rms,
        "cis_bend_rms_deg": cis_rms,
        "bond_angle_variance_deg2": bond_angle_variance,
        "baur_distortion": baur,
        "radius_pair_range": normalized_metrics["pair_range"],
        "radius_uniaxial_polarity": normalized_metrics["uniaxial_polarity"],
        "strain_bend_correlation": strain_bend_correlation,
        "donor_size_bond_correlation": donor_size_bond_correlation,
        "donor_plane_twist_mean_deg": plane_twist_mean,
    }
    pair_keys = [(keys[i], keys[j]) for i, j in matching]
    diagnostics = {
        "trans_rms_gate_deg": settings.trans_rms_max_deg,
        "cis_rms_gate_deg": settings.cis_rms_max_deg,
        "cis_angle_count": len(cis_angles),
        "bond_angle_variance_denominator": len(cis_angles) - 1,
        "donor_size_mismatch_pair_count": donor_mismatch_pairs,
        "donor_plane_twist_pair_count": len(plane_twists),
        "matching_objective_sum_sq_deg2": best[0][0],
    }
    return SiteResult(
        center_index=center_index,
        center_symbol=center_symbol,
        neighbor_keys=keys,
        neighbor_symbols=symbols,
        pair_indices=list(matching),
        pair_neighbor_keys=pair_keys,
        pair_mean_lengths=[float(x) for x in pair_means],
        pair_trans_bends_deg=[float(x) for x in trans_bends],
        short_pair_axis=[float(x) for x in short_axis],
        values=values,
        diagnostics=diagnostics,
    )


