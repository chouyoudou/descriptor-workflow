# Periodic host--guest geometry (PHGG): semantics and provenance

**Feature ID:** `periodic-host-guest-geometry/1`  
**Research seed:** DOI `10.1002/advs.202409093`  
**Implementation:** standard-library Python; fixed 18-scalar output; VASP 5/6
POSCAR input.

## Scope boundary

PHGG is a static structure descriptor for three questions only:

1. Which radius-gated connected components extend periodically, and what is
   the rank (0--3) of their independent lattice translations?
2. How much host-only open space is seen on a deterministic periodic grid?
3. When finite components are explicitly present, what is their center/surface
   geometry relative to the periodic host?

It does **not** estimate energetic content, detonation, impact or friction
sensitivity, synthesis conditions, stability, adsorption, diffusion,
transport, charge, oxidation state, bond energy, or material optimization.
The structure itself must contain every atom to be analyzed; no missing guest,
hydrogen, occupancy, or solvent is inferred.

## Paper-derived structural inspiration

The source article reports a six-coordinate metal environment that is bridged
into a two-dimensional layer, an approximately parallelogram-shaped window
containing an ordered but uncoordinated counterion, and a shortest metal--guest
distance used to reject coordination. Across related structures it compares
coordination-bond spreads and pore sizes, and it separately discusses explicit
hydrogen-bond and pi-contact environments. PHGG retains only the transferable
structural questions: periodic connectivity, host-only open-space scale, and
explicit host/guest contact geometry. It does not encode the article's
application targets or performance measurements.

The article is open access under a Creative Commons Attribution (CC BY) license. No article figure, table, or source
text is incorporated into the implementation.

## Established methods and aliases checked

### Periodic connected-component dimensionality

Larsen, Pandey, Strange, and Jacobsen, *Physical Review Materials* 3, 034003
(2019), DOI `10.1103/PhysRevMaterials.3.034003`, identify spatially connected
components and assign dimensional character. Mature software aliases include
ASE's dimensionality analysis and pymatgen's `get_dimensionality_larsen`.

PHGG independently implements the exact integer rank of quotient-graph cycle
translations for one fixed radius-gated graph. It does **not** reproduce the
Larsen score as a function of a continuously varied bond parameter, and it does
not copy ASE or pymatgen source.

### Pore geometry

Zeo++ (Willems et al., *Microporous and Mesoporous Materials* 149, 134--141,
2012, DOI `10.1016/j.micromeso.2011.08.020`) uses a periodic Voronoi network.
Common output aliases are the largest included sphere (`Di`, often called LCD)
and largest free sphere (`Df`, often called PLD), with a third included-along-
free-path diameter in some interfaces.

PHGG's `host_largest_grid_included_diameter_A` is **not** Zeo++ `Di/LCD`, and
PHGG provides no `Df/PLD`. It is a finite-grid maximum of host-surface
clearance. No Zeo++ or Voro++ code is copied, linked, or required.

### Element radii

Bond perception uses the crystallographic covalent radii from Cordero et al.,
*Covalent radii revisited*, Dalton Transactions (2008), DOI
`10.1039/B801115J`. The paper derives radii from crystallographic data for most
elements through atomic number 96.

Non-bonded exclusion radii are transcribed from the `Van der waals radius`
column of pymatgen-core's
`dev_scripts/periodic_table_resources/radii.csv` at commit
`adf41cdff585538dcc3f647c3893455365bc9429`. That upstream data file and
repository are MIT-licensed. PHGG records every element that uses its explicit
fallback. It never substitutes ionic radii or guesses oxidation states.

## PHGG definitions

Let fractional positions be \(f_i\), lattice matrix \(L\), and integer image
translation \(n\in\mathbb{Z}^3\). Distances are evaluated with an exact finite
closest-image search under the supplied triclinic cell:

\[
 d_{ij}=\min_n \lVert (f_j-f_i+n)L\rVert.
\]

### 1. Radius-gated periodic multigraph

An edge is present when

\[
 d_{ij}(n) \le s_b(r_i^{cov}+r_j^{cov})+\delta_b,
\]

with defaults \(s_b=1.18\) and \(\delta_b=0.10\) Å. Image translations are
stored on edges. For each connected component, a spanning traversal assigns
integer image offsets. Every non-zero residual around a quotient-graph cycle
is a lattice translation; the component dimension is the integer rank of
those translations.

All components with rank > 0 are host. This keeps disconnected or
interpenetrated periodic nets together. If no periodic component exists, all
finite components tied for the largest atom count are a documented fallback
host; only smaller components are guests. The tie rule prevents host/guest
roles from depending on atom order.

### 2. Host-only grid clearance

For a grid point \(x\),

\[
 c(x)=\min_{i\in host}[d(x,i)-r_i^{surf}].
\]

The grid phase is chosen from a translation-covariant canonical host signature,
so translating or permuting atoms does not change the fixed scalars. Grid
counts are determined from lattice-vector lengths and capped by
`max_grid_points`.

* `host_largest_grid_included_diameter_A = 2 max(0, max_x c(x))`
* `host_grid_void_fraction_ratio` is the fraction with
  \(c(x)>r_{probe}\)
* `host_grid_mean_positive_clearance_A` is the mean of positive \(c(x)\)

These are resolution-dependent approximations. Grid dimensions and realized
axis steps are emitted as metadata.

### 3. Explicit host/guest contacts

For each guest atom, PHGG records the nearest center distance, minimum surface
gap, and minimum normalized surface distance

\[
 q_{ij}=d_{ij}/(r_i^{surf}+r_j^{surf}).
\]

The atom is counted as in contact when \(\min_j q_{ij}\le s_c\), with default
\(s_c=1.05\). This atom fraction is not a Hirshfeld-surface contact fraction.
All exactly tied nearest normalized contacts are retained for directionality,
so symmetric cages do not acquire an atom-index bias.

For each finite guest component, coordinates are unwrapped through its graph.
With unwrapped centroid \(g\),

\[
 R_{cavity}=\max(0,\min_{i\in host}[d(g,i)-r_i^{surf}]),
\]

\[
 R_{envelope}=\max_{k\in guest}[\lVert r_k-g\rVert+r_k^{surf}],
\]

and `guest_cavity_fit_ratio = R_envelope / R_cavity` when the cavity radius is
positive. Component values are weighted by guest atom count.

Directionality uses the second moment \(M\) of nearest contact unit vectors,
first averaging ties per guest atom and then averaging guest atoms:

\[
 A=\sqrt{\frac{3}{2}\lVert M-I/3\rVert_F^2}.
\]

`A=0` is isotropic and `A=1` is perfectly axial.

## Fixed scalar schema

The output order is fixed in `FEATURE_NAMES`:

1. `component_count_per_cell`
2. `host_network_dimensionality`
3. `host_selection_periodic_ratio`
4. `host_atom_fraction_ratio`
5. `finite_guest_component_count_per_cell`
6. `finite_guest_atom_fraction_ratio`
7. `host_guest_present_ratio`
8. `host_largest_grid_included_diameter_A`
9. `host_grid_void_fraction_ratio`
10. `host_grid_mean_positive_clearance_A`
11. `host_guest_min_distance_A`
12. `host_guest_min_surface_gap_A`
13. `host_guest_mean_nearest_surface_gap_A`
14. `host_guest_contact_atom_fraction_ratio`
15. `guest_centered_cavity_diameter_A`
16. `guest_surface_envelope_diameter_A`
17. `guest_cavity_fit_ratio`
18. `guest_contact_directionality_anisotropy_ratio`

When no finite guest is present, guest-only channels are typed numeric zero;
`host_guest_present_ratio` and metadata validity fields carry applicability.

## Original method, treatment, and new definitions

| Category | Item |
| --- | --- |
| Established scientific method/alias | Connected-component dimensionality and Larsen/ASE/pymatgen terminology. |
| Established scientific method/alias | Zeo++ Voronoi pore terms `Di/LCD` and `Df/PLD`, used only for contrast. |
| Established lookup | Cordero covalent radii; pymatgen-core van der Waals radius table. |
| PHGG treatment | Fixed radius gate; exact closest image; all-periodic host partition; canonical finite grid. |
| PHGG-inspired new scalar | Guest-centered cavity/envelope ratio. |
| PHGG-inspired new scalar | Guest-normalized tied-contact directionality anisotropy. |

## Licensing

The PHGG implementation is MIT-licensed in
`descriptors/periodic_host_guest_geometry.LICENSE`. The same file reproduces
the required pymatgen-core MIT notice for the transcribed non-bonded radius
data. Scientific articles are cited but their prose, figures, and algorithm
source are not copied. Zeo++ is not a dependency and none of its source is
incorporated.
