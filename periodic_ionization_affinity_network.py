"""Periodic ionization-affinity network descriptors from an explicit structure.

The fixed descriptor uses only an ordered periodic structure and a frozen neutral-
atom lookup table. It does not infer oxidation state, charge transfer, band gap,
formation energy, thermodynamic stability, synthesis history, or device response.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from pian_geometry import _adaptive_shell, _fabric, _nearest_image_vector, _stats
from pian_model import (
    DEFAULT_TABLE,
    DescriptorUnavailable,
    ElementRecord,
    SHELL_FACTOR,
    StructureData,
    TABLE_SCHEMA,
    _canonical_table_path,
    load_element_table,
    normalize_structure,
    parse_poscar_text,
    structure_from_poscar,
    structure_from_record,
    table_sha256,
)

SCHEMA = "periodic-ionization-affinity-network/1"
FEATURE_NAMES = (
    "ie1_site_mean_ev", "ie1_site_std_ev", "ie1_site_range_ev",
    "electron_affinity_site_mean_ev", "electron_affinity_site_std_ev", "electron_affinity_site_range_ev",
    "first_shell_coordination_mean", "first_shell_coordination_std", "first_shell_distance_mean_angstrom", "first_shell_distance_cv",
    "ie1_edge_abs_contrast_mean_ev", "ie1_edge_abs_contrast_std_ev", "ie1_edge_abs_contrast_max_ev",
    "electron_affinity_edge_abs_contrast_mean_ev", "electron_affinity_edge_abs_contrast_std_ev", "electron_affinity_edge_abs_contrast_max_ev",
    "neutral_transfer_gap_edge_mean_ev", "neutral_transfer_gap_edge_min_ev", "neutral_transfer_gap_edge_std_ev",
    "hardness_edge_abs_contrast_mean_ev", "hardness_edge_abs_contrast_std_ev", "hardness_edge_abs_contrast_max_ev",
    "chi_contrast_shell_asymmetry_mean", "chi_contrast_shell_asymmetry_std",
    "chi_contrast_fabric_q2", "chi_contrast_fabric_linearity", "chi_contrast_fabric_planarity", "chi_contrast_fabric_isotropy",
)
FEATURE_UNITS = {name: "eV" if name.endswith("_ev") else "angstrom" if name.endswith("_angstrom") else "dimensionless" for name in FEATURE_NAMES}


def _set_null(features: dict[str,float|None], null_reasons: dict[str,dict[str,str]], names: Iterable[str], *, category: str, code: str, message: str) -> None:
    for name in names:
        features[name] = None
        null_reasons[name] = {"category":category, "code":code, "message":message}


def compute_descriptor(structure: StructureData, *, element_table_path: str|Path = DEFAULT_TABLE) -> dict[str,Any]:
    if not isinstance(structure, StructureData):
        raise DescriptorUnavailable("input", "not_structure_data", "input must be StructureData")
    table=load_element_table(element_table_path); records:list[ElementRecord]=[]; unknown=[]
    for symbol in structure.species:
        record=table.get(symbol)
        if record is None: unknown.append(symbol)
        else: records.append(record)
    if unknown:
        raise DescriptorUnavailable("lookup", "unknown_element_symbol", f"element table has no rows for {sorted(set(unknown))}")
    shell=_adaptive_shell(structure); centers=shell["center"]; points=shell["point"]; distances=shell["distances"]; vectors=shell["vectors"]; coordination=shell["coordination"].astype(np.float64)
    features:dict[str,float|None]={name:None for name in FEATURE_NAMES}; null_reasons:dict[str,dict[str,str]]={}
    ie1_complete=all(item.first_ionization_energy_ev is not None for item in records)
    ea_complete=all(item.electron_affinity_ev is not None for item in records)
    if not ie1_complete:
        missing=sorted({item.symbol for item in records if item.first_ionization_energy_ev is None})
        raise DescriptorUnavailable("lookup", "missing_first_ionization_energy", f"first ionization energy missing for {missing}")
    ie1=np.asarray([float(item.first_ionization_energy_ev) for item in records])
    mean,std,lo,hi=_stats(ie1); features["ie1_site_mean_ev"]=mean; features["ie1_site_std_ev"]=std; features["ie1_site_range_ev"]=hi-lo
    mean,std,_,_=_stats(coordination); features["first_shell_coordination_mean"]=mean; features["first_shell_coordination_std"]=std
    mean,std,_,_=_stats(distances); features["first_shell_distance_mean_angstrom"]=mean; features["first_shell_distance_cv"]=std/mean
    edge=np.abs(ie1[centers]-ie1[points]); mean,std,_,hi=_stats(edge)
    features["ie1_edge_abs_contrast_mean_ev"]=mean; features["ie1_edge_abs_contrast_std_ev"]=std; features["ie1_edge_abs_contrast_max_ev"]=hi
    chi_defined=0
    ea_names=("electron_affinity_site_mean_ev","electron_affinity_site_std_ev","electron_affinity_site_range_ev","electron_affinity_edge_abs_contrast_mean_ev","electron_affinity_edge_abs_contrast_std_ev","electron_affinity_edge_abs_contrast_max_ev","neutral_transfer_gap_edge_mean_ev","neutral_transfer_gap_edge_min_ev","neutral_transfer_gap_edge_std_ev","hardness_edge_abs_contrast_mean_ev","hardness_edge_abs_contrast_std_ev","hardness_edge_abs_contrast_max_ev","chi_contrast_shell_asymmetry_mean","chi_contrast_shell_asymmetry_std","chi_contrast_fabric_q2","chi_contrast_fabric_linearity","chi_contrast_fabric_planarity","chi_contrast_fabric_isotropy")
    if not ea_complete:
        missing=sorted({item.symbol for item in records if item.electron_affinity_ev is None})
        _set_null(features,null_reasons,ea_names,category="lookup",code="missing_electron_affinity",message=f"electron affinity missing for {missing}")
    else:
        ea=np.asarray([float(item.electron_affinity_ev) for item in records])
        mean,std,lo,hi=_stats(ea); features["electron_affinity_site_mean_ev"]=mean; features["electron_affinity_site_std_ev"]=std; features["electron_affinity_site_range_ev"]=hi-lo
        edge=np.abs(ea[centers]-ea[points]); mean,std,_,hi=_stats(edge)
        features["electron_affinity_edge_abs_contrast_mean_ev"]=mean; features["electron_affinity_edge_abs_contrast_std_ev"]=std; features["electron_affinity_edge_abs_contrast_max_ev"]=hi
        gap=ie1[centers]-ea[points]; mean,std,lo,_=_stats(gap)
        features["neutral_transfer_gap_edge_mean_ev"]=mean; features["neutral_transfer_gap_edge_min_ev"]=lo; features["neutral_transfer_gap_edge_std_ev"]=std
        hardness=0.5*(ie1-ea); edge=np.abs(hardness[centers]-hardness[points]); mean,std,_,hi=_stats(edge)
        features["hardness_edge_abs_contrast_mean_ev"]=mean; features["hardness_edge_abs_contrast_std_ev"]=std; features["hardness_edge_abs_contrast_max_ev"]=hi
        chi=0.5*(ie1+ea); weights=np.abs(chi[centers]-chi[points])
        chi_names=("chi_contrast_shell_asymmetry_mean","chi_contrast_shell_asymmetry_std","chi_contrast_fabric_q2","chi_contrast_fabric_linearity","chi_contrast_fabric_planarity","chi_contrast_fabric_isotropy")
        if float(np.sum(weights)) <= 1e-15:
            _set_null(features,null_reasons,chi_names,category="definition_domain",code="zero_chi_contrast",message="all first-shell Mulliken-chi lookup contrasts are zero")
        else:
            asymmetry=[]
            for site in range(len(structure.species)):
                mask=centers==site; local=weights[mask]; total=float(np.sum(local))
                if total <= 1e-15: continue
                unit=vectors[mask]/distances[mask,None]
                asymmetry.append(float(np.linalg.norm(np.sum(local[:,None]*unit,axis=0))/total))
            if not asymmetry:
                raise DescriptorUnavailable("algorithm","chi_asymmetry_coverage","global chi contrast was nonzero but no site had positive weight")
            chi_defined=len(asymmetry); array=np.asarray(asymmetry)
            features["chi_contrast_shell_asymmetry_mean"]=float(np.mean(array)); features["chi_contrast_shell_asymmetry_std"]=float(np.std(array))
            fabric=_fabric(vectors,weights)
            for suffix in ("q2","linearity","planarity","isotropy"): features[f"chi_contrast_fabric_{suffix}"]=fabric[suffix]
    for name,value in features.items():
        if value is not None:
            number=float(value)
            if not math.isfinite(number): raise DescriptorUnavailable("numerical","non_finite_feature",f"{name} is not finite")
            features[name]=0.0 if abs(number)<1e-15 else number
    if tuple(features) != FEATURE_NAMES:
        raise DescriptorUnavailable("algorithm","feature_order_mismatch","fixed feature order changed")
    return {"schema":SCHEMA,"features":features,"feature_units":FEATURE_UNITS,"null_reasons":null_reasons,"metadata":{"site_count":len(structure.species),"directed_first_shell_edge_count":int(len(centers)),"shell_factor":SHELL_FACTOR,"element_table_schema":TABLE_SCHEMA,"element_table_file":Path(element_table_path).name,"element_table_sha256":table_sha256(_canonical_table_path(element_table_path)),"chi_asymmetry_defined_site_count":chi_defined,"lookup_complete_ie1":ie1_complete,"lookup_complete_ea":ea_complete,"target_properties_read":False}}


def compute_from_record(record: Mapping[str,Any], *, element_table_path: str|Path=DEFAULT_TABLE) -> dict[str,Any]:
    return compute_descriptor(structure_from_record(record), element_table_path=element_table_path)


def _main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--poscar"); source.add_argument("--record-json"); parser.add_argument("--table",default=str(DEFAULT_TABLE)); parser.add_argument("--vasp4-symbols"); parser.add_argument("--pretty",action="store_true"); args=parser.parse_args()
    try:
        if args.poscar:
            structure=structure_from_poscar(args.poscar,vasp4_symbols=args.vasp4_symbols.split(",") if args.vasp4_symbols else None)
        else:
            structure=structure_from_record(json.loads(Path(args.record_json).read_text(encoding="utf-8")))
        print(json.dumps(compute_descriptor(structure,element_table_path=args.table),indent=2 if args.pretty else None,sort_keys=False,allow_nan=False)); return 0
    except DescriptorUnavailable as exc:
        print(json.dumps({"status":"unavailable","reason":exc.as_dict()},allow_nan=False)); return 2


if __name__ == "__main__":
    raise SystemExit(_main())
