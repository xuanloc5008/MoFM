"""
Topology utilities: Betti numbers, Persistence Diagrams via GUDHI,
Wasserstein distance, and topology-preserving augmentation filter.
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.ndimage import label as scipy_label

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("WARNING: gudhi not installed. Topology features disabled.")


# ---- Betti Numbers ----

def compute_betti_numbers(mask: np.ndarray) -> Tuple[int, int]:
    """Compute (beta0, beta1) for a 2D binary mask.
    beta0 = number of connected components
    beta1 = number of holes (loops)
    Uses Euler characteristic: chi = beta0 - beta1, chi = V - E + F (cubical)
    """
    if mask.sum() == 0:
        return (0, 0)

    # beta0: count connected components
    labeled, beta0 = scipy_label(mask)

    # beta1 via cubical complex
    if GUDHI_AVAILABLE:
        # Superlevel filtration on the binary mask
        cc = gudhi.CubicalComplex(
            dimensions=mask.shape,
            top_dimensional_cells=(1.0 - mask.astype(float)).flatten()
        )
        cc.persistence()
        betti = cc.betti_numbers()
        beta1 = betti[1] if len(betti) > 1 else 0
    else:
        # Fallback: Euler characteristic
        # chi = beta0 - beta1 for 2D
        # Estimate chi from pixel adjacency
        V = mask.sum()
        E_h = np.sum(mask[:, :-1] & mask[:, 1:])
        E_v = np.sum(mask[:-1, :] & mask[1:, :])
        F = np.sum(mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:])
        chi = V - (E_h + E_v) + F
        beta1 = max(0, beta0 - chi)

    return (beta0, beta1)


def compute_all_betti(seg: np.ndarray, class_map: Dict[str, int] = None
                      ) -> Dict[str, Tuple[int, int]]:
    """Compute Betti numbers for each cardiac structure in a segmentation.
    Args:
        seg: (H, W) integer segmentation, classes: 0=BG, 1=LV, 2=Myo, 3=RV
        class_map: mapping from structure name to class index
    Returns:
        Dict mapping structure name to (beta0, beta1)
    """
    if class_map is None:
        class_map = {"LV": 1, "Myo": 2, "RV": 3}

    result = {}
    for name, idx in class_map.items():
        mask = (seg == idx).astype(np.uint8)
        result[name] = compute_betti_numbers(mask)
    return result


def check_betti_violations(betti: Dict[str, Tuple[int, int]],
                           targets: Dict[str, Tuple[int, int]] = None
                           ) -> Dict[str, Tuple[int, int]]:
    """Check which structures violate their Betti-number invariants.
    Returns dict of violations: structure -> (actual, target).
    """
    if targets is None:
        targets = {"LV": (1, 0), "Myo": (1, 1), "RV": (1, 0)}

    violations = {}
    for name, target in targets.items():
        if name in betti and betti[name] != target:
            violations[name] = (betti[name], target)
    return violations


# ---- Persistence Diagrams ----

def compute_persistence_diagram(mask: np.ndarray, prob_map: np.ndarray = None,
                                homology_dim: int = None
                                ) -> List[Tuple[float, float, int]]:
    """Compute persistence diagram for a scalar field.
    Args:
        mask: binary mask (used if prob_map is None)
        prob_map: probability map (continuous, for finer PD)
        homology_dim: if specified, filter to this dimension only
    Returns:
        List of (birth, death, dimension) tuples
    """
    if not GUDHI_AVAILABLE:
        return []

    field = prob_map if prob_map is not None else mask.astype(float)

    # Superlevel set filtration (negate for sublevel)
    cc = gudhi.CubicalComplex(
        dimensions=field.shape,
        top_dimensional_cells=(1.0 - field).flatten()
    )
    cc.persistence()
    pd = []
    for dim, (b, d) in cc.persistence():
        if d == float('inf'):
            d = 1.0  # Cap infinite death
        # Convert from sublevel to superlevel
        birth_sup = 1.0 - b
        death_sup = 1.0 - d
        if homology_dim is None or dim == homology_dim:
            pd.append((birth_sup, death_sup, dim))
    return pd


def wasserstein_distance_pd(pd1: List[Tuple[float, float, int]],
                            pd2: List[Tuple[float, float, int]],
                            p: int = 1, dim: int = None
                            ) -> float:
    """Approximate Wasserstein distance between two persistence diagrams.
    Uses gudhi.wasserstein if available, else simple matching.
    """
    def _filter_dim(pd, d):
        return [(b, death) for b, death, dim_ in pd if d is None or dim_ == d]

    pts1 = _filter_dim(pd1, dim)
    pts2 = _filter_dim(pd2, dim)

    if not pts1 and not pts2:
        return 0.0

    if GUDHI_AVAILABLE:
        try:
            from gudhi.wasserstein import wasserstein_distance as gw_dist
            a1 = np.array(pts1) if pts1 else np.empty((0, 2))
            a2 = np.array(pts2) if pts2 else np.empty((0, 2))
            return gw_dist(a1, a2, order=p)
        except ImportError:
            pass

    # Fallback: greedy matching
    from scipy.spatial.distance import cdist
    if not pts1:
        return sum(abs(b - d) for b, d in pts2)
    if not pts2:
        return sum(abs(b - d) for b, d in pts1)

    a1 = np.array(pts1)
    a2 = np.array(pts2)
    dists = cdist(a1, a2, metric='chebyshev')

    total = 0.0
    used1, used2 = set(), set()
    flat = [(dists[i, j], i, j) for i in range(len(a1)) for j in range(len(a2))]
    flat.sort()
    for cost, i, j in flat:
        if i not in used1 and j not in used2:
            total += cost ** p
            used1.add(i)
            used2.add(j)
    # Unmatched → match to diagonal
    for i in range(len(a1)):
        if i not in used1:
            total += (abs(a1[i, 0] - a1[i, 1]) / 2) ** p
    for j in range(len(a2)):
        if j not in used2:
            total += (abs(a2[j, 0] - a2[j, 1]) / 2) ** p
    return total ** (1.0 / p)


def pd_similarity(pd1, pd2, dim=None) -> float:
    """PD similarity: Sim_PD = 1 / (1 + W1(PD1, PD2))."""
    w = wasserstein_distance_pd(pd1, pd2, p=1, dim=dim)
    return 1.0 / (1.0 + w)


# ---- TPA Filter ----

def check_topology_preserved(mask: np.ndarray,
                             targets: Dict[str, Tuple[int, int]] = None
                             ) -> bool:
    """Check if a segmentation mask preserves mid-ventricular Betti invariants.
    Used as the TPA acceptance filter.
    """
    betti = compute_all_betti(mask)
    violations = check_betti_violations(betti, targets)
    return len(violations) == 0


# ---- Persistence Lifetime Ratio ----

def compute_plr(pd: List[Tuple[float, float, int]],
                violating_feature_idx: int = -1) -> float:
    """Persistence Lifetime Ratio for TTEC Signal 2.
    PLR = lifetime_violating / median(lifetime_stable)
    """
    lifetimes = [abs(b - d) for b, d, _ in pd]
    if not lifetimes:
        return 0.0

    lifetimes_sorted = sorted(lifetimes, reverse=True)
    violating_lt = lifetimes_sorted[min(violating_feature_idx, len(lifetimes_sorted) - 1)]

    # Stable = long-lived features (top 50%)
    n_stable = max(1, len(lifetimes_sorted) // 2)
    stable_median = np.median(lifetimes_sorted[:n_stable])

    if stable_median == 0:
        return float('inf')
    return violating_lt / stable_median
