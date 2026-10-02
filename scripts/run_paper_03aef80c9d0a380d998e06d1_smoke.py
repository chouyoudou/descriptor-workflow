#!/usr/bin/env python3
"""Deterministic 1000-POSCAR public qualification."""
from __future__ import annotations
import argparse, hashlib, json, math, platform
from pathlib import Path
import numpy as np
from molecular_packing_descriptor import DESCRIPTOR_NAMES, analyze_structure, parse_poscar_text

TASK = "paper-03aef80c9d0a380d998e06d1"
BOUNDED = set(DESCRIPTOR_NAMES) - {"coordination_shell_radial_distortion", "interfragment_contact_coordination"}


def poscar(species, xyz, cell, comment, order=None):
    order = order or list(dict.fromkeys(species))
    ids = [i for s in order for i, x in enumerate(species) if x == s]
    counts = [species.count(s) for s in order]
    frac = np.asarray(xyz)[ids] @ np.linalg.inv(cell)
    lines = [comment, "1.0", *(" ".join(f"{x:.12f}" for x in row) for row in cell),
             " ".join(order), " ".join(map(str, counts)), "Direct",
             *(" ".join(f"{x:.12f}" for x in row) for row in frac)]
    return "\n".join(lines) + "\n"


def values(species, xyz, cell, comment="case", order=None):
    return analyze_structure(parse_poscar_text(poscar(species, xyz, cell, comment, order))).values


def case(i):
    cell = np.diag([24.0, 25.0, 26.0])
    k = i % 5
    if k == 0:
        a = 0.05 + 0.55 * ((i % 17) / 16)
        pts = [[4.0, 7.0, 7.0]]
        heading = 0.0
        for j in range(1, 6 + i % 3):
            heading += a * math.sin(i + j)
            pts.append(np.asarray(pts[-1]) + [1.45 * math.cos(heading), 1.45 * math.sin(heading), 0.0])
        return "chain", ["C"] * len(pts), np.asarray(pts), cell
    if k == 1:
        center = np.array([12.0, 12.0, 12.0])
        u = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float) / math.sqrt(3)
        d = 2.25 * (1 + 0.04 * np.array([math.sin(i+j) for j in range(4)]))
        return "coordination", ["Fe"] + ["Cl"] * 4, np.vstack([center, center + u*d[:,None]]), cell
    if k == 2:
        d = 2.55 + 0.45 * ((i % 19) / 18)
        return "packing", ["H", "Cl"], np.array([[8,8,8],[8+d,8,8]], float), cell
    if k == 3:
        c = np.array([12.0, 12.0, 12.0]); u = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
        return "mixed", ["C"] + ["O"]*6, np.vstack([c, c + 3.0*u]), cell
    return "fallback", ["He"], np.array([[5.0,5.0,5.0]]), cell


def check(v):
    assert tuple(v) == DESCRIPTOR_NAMES
    for name, x in v.items():
        assert type(x) is float and math.isfinite(x) and x >= -1e-12
        if name in BOUNDED: assert x <= 1 + 1e-10


def same(a, b, tol=3e-8):
    for name in DESCRIPTOR_NAMES: assert math.isclose(a[name], b[name], rel_tol=tol, abs_tol=tol), name


def mechanisms():
    cell = np.diag([24.0]*3)
    straight = np.array([[4+1.5*i,6,6] for i in range(6)], float)
    bent = np.array([[4,6,6],[5.5,6,6],[6.8,6.75,6],[6.8,8.25,6],[5.5,9,6],[4,9,6]], float)
    s, b = values(["C"]*6, straight, cell), values(["C"]*6, bent, cell)
    c = np.array([12.0]*3); u = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)/math.sqrt(3)
    r = values(["Fe"]+["Cl"]*4, np.vstack([c,c+2.25*u]), cell)
    d = values(["Fe"]+["Cl"]*4, np.vstack([c,c+u*np.array([2.0,2.2,2.45,2.65])[:,None]]), cell)
    near = values(["H","Cl"], np.array([[8,8,8],[10.55,8,8]], float), cell)
    far = values(["H","Cl"], np.array([[8,8,8],[11.05,8,8]], float), cell)
    dirs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
    iso = values(["C"]+["O"]*6, np.vstack([c,c+3*dirs]), cell)
    out = {
      "straight_spacer_more_extended": s["spacer_path_extension"] > b["spacer_path_extension"] + .15,
      "straight_component_more_rodlike": s["component_shape_anisotropy"] > b["component_shape_anisotropy"],
      "distortion_increases_radial_cv": d["coordination_shell_radial_distortion"] > r["coordination_shell_radial_distortion"] + .05,
      "closer_contact_more_compact": near["interfragment_contact_compactness"] > far["interfragment_contact_compactness"],
      "linear_contacts_more_anisotropic": near["interfragment_contact_anisotropy"] > iso["interfragment_contact_anisotropy"] + .9,
    }
    assert all(out.values()), out
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--n-cases", type=int, default=1000); p.add_argument("--seed", type=int, default=20261002); p.add_argument("--output", required=True); a = p.parse_args()
    assert a.n_cases >= 1000
    rng = np.random.default_rng(a.seed); digest = hashlib.sha256(); passed = 0; failures = []; counts = {}; superchecks = 0
    for i in range(a.n_cases):
        scenario, species, xyz, cell = case(i); counts[scenario] = counts.get(scenario, 0) + 1
        try:
            v = values(species, xyz, cell, f"qualification-{i}"); check(v)
            order = list(reversed(list(dict.fromkeys(species))))
            moved = values(species, xyz + rng.uniform(-3,3,3), cell, "translated", order); same(v, moved)
            if i % 20 == 0:
                sc = cell.copy(); sc[0] *= 2
                sv = values(species*2, np.vstack([xyz, xyz+cell[0]]), sc, "2x1x1"); same(v, sv, 5e-8); superchecks += 1
            digest.update((json.dumps(v, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode())
            passed += 1
        except Exception as e:
            if len(failures) < 25: failures.append({"index": i, "scenario": scenario, "error": f"{type(e).__name__}: {e}"})
    result = {"schema":"descriptor-thousand-example-validation/v1","task_id":TASK,"seed":a.seed,"n_total":a.n_cases,"n_pass":passed,"n_fail":a.n_cases-passed,"n_skipped":0,"pass_rate":passed/a.n_cases,"scenario_counts":dict(sorted(counts.items())),"invariance_checks":{"translation_and_species_order":passed,"2x1x1_supercell":superchecks},"mechanism_checks":mechanisms(),"descriptor_names":list(DESCRIPTOR_NAMES),"row_sha256":digest.hexdigest(),"software_versions":{"python":platform.python_version(),"numpy":np.__version__},"failure_examples":failures}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(result, sort_keys=True)); raise SystemExit(0 if result["n_fail"] == 0 else 1)

if __name__ == "__main__": main()
