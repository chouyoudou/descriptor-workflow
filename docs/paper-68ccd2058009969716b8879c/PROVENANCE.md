# Provenance and method distinction

Task: `paper-68ccd2058009969716b8879c`  
DOI: `10.1038/s41598-023-37633-9`

## Paper source

The supplied article is:

H. Salmenjoki, S. Papanikolaou, D. Shi, D. Tourret,
C. M. Cepeda-Jiménez, M. T. Pérez-Prado, L. Laurson, and M. J. Alava,
"Machine learning dislocation density correlations and solute effects in Mg-based
alloys," *Scientific Reports* **13**, 11114 (2023).

The article studies pure Mg and Mg--Zn samples through EBSD-derived grain and GND
information, dimensionality reduction, support-vector prediction, and grain graph
networks. It says that its supporting data are available from the corresponding
author on reasonable request. The article is distributed under CC BY 4.0.

No EBSD image, supplementary table, sample label, experimental dislocation density,
strain/deformation label, learned parameter, trained network, or article figure is
copied into this implementation. The paper provides scientific motivation only.
The formulas and source code here are independently written for a different input
and claim: an explicit static periodic structure.

## Elemental reference

The only independently tabulated elemental value is neutral-element atomic number
`Z` supplied through `pymatgen.core.Element.Z`. Formal private qualification pins
pymatgen 2026.5.4. Atomic number is a factual element identifier; no paper-derived
or target-derived elemental table is embedded.

The implementation normalizes `abs(Z_i-Z_j)` by 117, the maximum possible difference
for atomic numbers 1 through 118. It rejects oxidation-state-labelled `Species`,
dummy species, and partial occupancy rather than silently converting them.

Pymatgen is distributed under the MIT License. The applicable upstream notice is
reproduced in `THIRD_PARTY-paper-68ccd2058009969716b8879c.md`. This repository does
not copy pymatgen source; it imports its documented public `Structure`, `Element.Z`,
POSCAR parser, and periodic neighbour-search API.

## Mature aliases and differences

### matminer `LocalPropertyDifference`

Matminer's mature `LocalPropertyDifference` computes elemental-property differences
between a site and neighbouring sites, normally with a chosen near-neighbour method.
The atomic-number contrast channel here belongs to that broad family and is not
claimed as a new concept. The present fixed descriptor differs in its tie-complete
12-neighbour shell, log-radial geometry channels, central-element variance
decomposition, composition-surprisal aggregate, and pairwise atomic-number-weighted
separation of element-conditioned means.

### matminer `SiteStatsFingerprint`

`SiteStatsFingerprint` is a general framework for aggregating site features into
structure features. The present implementation is a narrow, auditable fixed schema
with explicitly defined statistics and resource bounds. It is not represented as a
replacement for the broader matminer framework.

### Warren--Cowley short-range order

Warren--Cowley parameters compare observed pair probabilities with a random-mixing
reference, usually for specified species and neighbour shells. This implementation
does not assume a random alloy, infer occupancy probabilities, or emit pair-specific
SRO parameters. Its `z_weighted_*_separation` fields weight differences between
central-element-conditioned geometry means; they are not Warren--Cowley coefficients.

### ANOVA / intraclass variance decomposition

The reported between-element fractions use the ordinary decomposition of variance
into variation between categorical group means and residual variation within groups.
That statistical identity is mature. The contribution is the explicit choice of
site quantities, periodic shell, fixed aggregation, typed domain, and static-structure
scope, not invention of ANOVA.

### Existing repository first-shell directionality work

The repository's `intrinsic-first-shell-directionality/1` schema describes adaptive
first-shell coordination, off-centering, and rank-2 bond-fabric directionality,
including atomic-number-weighted fabric channels. The present descriptor deliberately
avoids a fabric tensor. It uses a tie-complete fixed-rank shell and asks a different
question: how much radial/chemical site heterogeneity is aligned with the central
element class, and how strongly element-conditioned means are separated.

### SOAP, ACSF, CrystalNN fingerprints, and learned graph embeddings

SOAP, atom-centred symmetry functions, CrystalNN fingerprints, and graph embeddings
can encode substantially richer local environments. They are appropriate comparison
baselines but are not aliases for this 30-scalar interpretable summary. No kernel,
learned embedding, response target, or training is used here.

## Naming audit

The name **element-conditioned local geometry** is descriptive and avoids reserved or
mechanistically stronger claims. In particular:

- `rarity_weighted_*` means a deterministic global-composition surprisal weight;
- it is not named `solute`, `dopant`, `defect`, `segregation`, or `dilute-limit`;
- `atomic_number_contrast` is not charge, valence, electronegativity, or electronic
  contrast;
- `environment_heterogeneity` is variation inside the supplied periodic cell, not a
  measured grain population or mesoscale microstructure;
- `knn_shell` is geometric, not a bond network.

## Reproducibility identity

The fixed schema name, ordered feature list, constants, typed errors, and source
identity are emitted with the formal run evidence. Public and private final science
source must be byte-identical. The thousand-structure execution reads only the frozen
POSCAR-equivalent records and explicitly records that no property target or paper
EBSD/dislocation label was read.
