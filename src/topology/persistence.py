"""
Topological Data Analysis Utilities
--------------------------------------
Computes Persistence Diagrams (PD) from 2-D images using sublevel filtration.
Implements Wasserstein W1 distance between diagrams for SimPD computation.

Primary libraries:
  - gudhi  : Cubical Complex + Persistence computation
  - ripser  : fast Rips complex (fallback)
  - POT    : Python Optimal Transport for Wasserstein distances
"""
import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
_IS_MAIN_PROCESS = mp.current_process().name == "MainProcess"

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    if _IS_MAIN_PROCESS:
        logger.warning("gudhi not installed; falling back to ripser for PD computation")

try:
    import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

try:
    import ot  # POT library
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False
    if _IS_MAIN_PROCESS:
        logger.warning("POT (Python Optimal Transport) not installed; "
                       "using scipy-based W1 approximation")


# ─────────────────────────────────────────────────────────────────────────────
#  Persistence Diagram computation
# ─────────────────────────────────────────────────────────────────────────────

PersistenceDiagram = List[Tuple[float, float]]   # list of (birth, death) pairs


def _diagram_persistences(pd: PersistenceDiagram) -> np.ndarray:
    """Return sorted persistence lengths for one diagram."""
    if len(pd) == 0:
        return np.empty((0,), dtype=np.float32)
    arr = np.asarray(pd, dtype=np.float32)
    pers = np.clip(arr[:, 1] - arr[:, 0], a_min=0.0, a_max=None)
    return np.sort(pers)[::-1]


def summarize_persistence_diagrams(
    diagrams: List[PersistenceDiagram],
    max_dim: int = 1,
    top_k: int = 8,
) -> np.ndarray:
    """
    Convert persistence diagrams into a fixed-width summary vector.

    For each homology dimension, we store:
      count, sum, mean, max, std, and the top-k persistence values.
    """
    features = []
    for dim in range(max_dim + 1):
        pers = _diagram_persistences(diagrams[dim] if dim < len(diagrams) else [])
        top_vals = np.zeros((top_k,), dtype=np.float32)
        n_keep = min(top_k, pers.shape[0])
        if n_keep > 0:
            top_vals[:n_keep] = pers[:n_keep]

        count = float(pers.shape[0])
        total = float(pers.sum()) if pers.size else 0.0
        mean = float(pers.mean()) if pers.size else 0.0
        max_v = float(pers.max()) if pers.size else 0.0
        std_v = float(pers.std()) if pers.size else 0.0
        features.extend([count, total, mean, max_v, std_v])
        features.extend(top_vals.tolist())
    return np.asarray(features, dtype=np.float32)


def compute_topology_vector(
    image: np.ndarray,
    max_dim: int = 1,
    threshold: float = 0.02,
    downsample_size: int = 64,
    top_k: int = 8,
) -> np.ndarray:
    """
    Compute a compact, fixed-width topology descriptor for one slice.

    The image is optionally downsampled before PH computation, then the
    resulting persistence diagrams are summarized into a feature vector that is
    cheap to cache on disk and cheap to compare on GPU during training.
    """
    from scipy.ndimage import zoom as ndimage_zoom

    img = np.squeeze(image).astype(np.float32, copy=False)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {img.shape}")

    scale = min(1.0, float(downsample_size) / float(max(img.shape)))
    if scale < 1.0:
        img = ndimage_zoom(img, scale, order=1)

    diagrams = compute_persistence_diagram(img, max_dim=max_dim, threshold=threshold)
    return summarize_persistence_diagrams(diagrams, max_dim=max_dim, top_k=top_k)


def compute_persistence_diagram(
    image: np.ndarray,
    max_dim: int = 1,
    threshold: float = 0.02,
) -> List[PersistenceDiagram]:
    """
    Compute persistence diagram for a 2-D image via sublevel-set filtration.

    Args:
        image    : 2-D numpy array (H, W), already normalised to [0, 1]
        max_dim  : maximum homology dimension (0 = components, 1 = loops)
        threshold: minimum persistence (death - birth) to include

    Returns:
        diagrams : list indexed by dimension [PD_H0, PD_H1]
                   each PD is a list of (birth, death) tuples
    """
    image_2d = np.squeeze(image)   # handle (1, H, W) input
    assert image_2d.ndim == 2, f"Expected 2-D image, got shape {image_2d.shape}"

    if GUDHI_AVAILABLE:
        return _compute_pd_gudhi(image_2d, max_dim, threshold)
    elif RIPSER_AVAILABLE:
        return _compute_pd_ripser(image_2d, max_dim, threshold)
    else:
        logger.warning("No TDA library available; returning empty diagrams")
        return [[] for _ in range(max_dim + 1)]


def _compute_pd_gudhi(
    image: np.ndarray, max_dim: int, threshold: float
) -> List[PersistenceDiagram]:
    """Gudhi CubicalComplex sublevel filtration."""
    cc = gudhi.CubicalComplex(top_dimensional_cells=image.flatten(),
                              dimensions=list(image.shape))
    cc.compute_persistence(homology_coeff_field=2, min_persistence=threshold)

    diagrams = [[] for _ in range(max_dim + 1)]
    for dim, (b, d) in cc.persistence():
        if dim <= max_dim:
            if d == float("inf"):
                d = image.max()
            diagrams[dim].append((float(b), float(d)))
    return diagrams


def _compute_pd_ripser(
    image: np.ndarray, max_dim: int, threshold: float
) -> List[PersistenceDiagram]:
    """
    Ripser-based fallback.  Downsample image to speed up computation,
    then run Rips complex on point cloud of pixel intensities.
    """
    # Downsample for speed
    from scipy.ndimage import zoom as ndimage_zoom
    scale = min(1.0, 32.0 / max(image.shape))
    small = ndimage_zoom(image, scale, order=1).astype(np.float32)
    H, W = small.shape

    # Point cloud: (pixel_value, row/H, col/W)
    rows, cols = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    pts = np.column_stack([small.ravel(), rows.ravel(), cols.ravel()])

    result = ripser.ripser(pts, maxdim=max_dim)
    diagrams = []
    for dim in range(max_dim + 1):
        pd = result["dgms"][dim]
        filtered = [
            (float(b), float(d))
            for b, d in pd
            if d != float("inf") and (d - b) >= threshold
        ]
        diagrams.append(filtered)
    return diagrams


# ─────────────────────────────────────────────────────────────────────────────
#  Wasserstein distance between diagrams
# ─────────────────────────────────────────────────────────────────────────────

def wasserstein_distance(
    pd1: PersistenceDiagram,
    pd2: PersistenceDiagram,
    p:   int = 1,
) -> float:
    """
    Compute W_p distance between two persistence diagrams.
    Uses POT library if available, otherwise a simple projection-based approx.
    Diagonal projections are handled explicitly.
    """
    if len(pd1) == 0 and len(pd2) == 0:
        return 0.0

    if POT_AVAILABLE:
        return _wasserstein_pot(pd1, pd2, p)
    else:
        return _wasserstein_scipy_approx(pd1, pd2, p)


def _pd_to_array(pd: PersistenceDiagram) -> np.ndarray:
    """Convert list of (birth, death) to Nx2 array."""
    if len(pd) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(pd, dtype=np.float32)


def _diagonal_projection(pts: np.ndarray) -> np.ndarray:
    """Project each (b, d) to its closest point on the diagonal."""
    mid = (pts[:, 0] + pts[:, 1]) / 2.0
    return np.column_stack([mid, mid])


def _wasserstein_pot(
    pd1: PersistenceDiagram,
    pd2: PersistenceDiagram,
    p:   int,
) -> float:
    """
    Full Wasserstein via POT's earth mover distance.
    Augments each diagram with diagonal projections of the other.
    """
    a1 = _pd_to_array(pd1)
    a2 = _pd_to_array(pd2)

    # Augment with diagonal points
    d1 = _diagonal_projection(a2) if len(a2) > 0 else np.empty((0, 2))
    d2 = _diagonal_projection(a1) if len(a1) > 0 else np.empty((0, 2))

    A = np.vstack([a1, d1]) if len(a1) > 0 else (d1 if len(d1) > 0 else np.zeros((1, 2)))
    B = np.vstack([a2, d2]) if len(a2) > 0 else (d2 if len(d2) > 0 else np.zeros((1, 2)))

    n, m = len(A), len(B)
    # Uniform weights
    wa = np.ones(n) / n
    wb = np.ones(m) / m

    # Cost matrix (L2 distance between 2-D points)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]   # n x m x 2
    M = np.linalg.norm(diff, axis=-1)
    if p == 2:
        M = M ** 2

    try:
        W = ot.emd2(wa, wb, M)
    except Exception:
        W = float(np.mean(M))

    return float(W ** (1.0 / p)) if p == 2 else float(W)


def _wasserstein_scipy_approx(
    pd1: PersistenceDiagram,
    pd2: PersistenceDiagram,
    p:   int,
) -> float:
    """
    Simple bottleneck-style approximation when POT is unavailable.
    Uses scipy linear_sum_assignment.
    """
    from scipy.optimize import linear_sum_assignment

    a1 = _pd_to_array(pd1)
    a2 = _pd_to_array(pd2)

    if len(a1) == 0:
        return float(np.sum([abs(d - b) / 2 for b, d in pd2]))
    if len(a2) == 0:
        return float(np.sum([abs(d - b) / 2 for b, d in pd1]))

    n, m = len(a1), len(a2)
    # Pad smaller diagram with diagonal projections
    if n < m:
        diag = _diagonal_projection(a1)
        a1 = np.vstack([a1, diag[:m - n]])
    elif m < n:
        diag = _diagonal_projection(a2)
        a2 = np.vstack([a2, diag[:n - m]])

    n = m = max(len(a1), len(a2))
    diff = a1[:, np.newaxis] - a2[np.newaxis, :]
    M = np.linalg.norm(diff, axis=-1)

    ri, ci = linear_sum_assignment(M)
    return float(M[ri, ci].sum())


# ─────────────────────────────────────────────────────────────────────────────
#  SimPD similarity score
# ─────────────────────────────────────────────────────────────────────────────

def sim_pd(
    pd1: List[PersistenceDiagram],
    pd2: List[PersistenceDiagram],
    dims: List[int] = (0, 1),
    p:   int = 1,
) -> float:
    """
    Topological similarity as defined in the paper:
        SimPD(x1, x2) = 1 / (1 + W1(PD*(x1), PD*(x2)))

    Aggregates Wasserstein distance over specified homology dimensions.
    """
    total_w = 0.0
    for dim in dims:
        pd_a = pd1[dim] if dim < len(pd1) else []
        pd_b = pd2[dim] if dim < len(pd2) else []
        total_w += wasserstein_distance(pd_a, pd_b, p)

    return 1.0 / (1.0 + total_w)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch topology for mini-batch contrastive learning
# ─────────────────────────────────────────────────────────────────────────────

def compute_batch_sim_matrix(
    images: np.ndarray,           # (B, 1, H, W)
    max_dim: int   = 1,
    threshold: float = 0.02,
) -> np.ndarray:
    """
    Compute B×B SimPD matrix for a mini-batch.
    Downsamples images to 64x64 for speed.
    Returns: (B, B) float32 similarity matrix.
    """
    from scipy.ndimage import zoom as ndimage_zoom

    B = images.shape[0]
    pds = []
    for i in range(B):
        img = np.squeeze(images[i])
        # Downsample for efficiency during training
        scale = min(1.0, 64.0 / max(img.shape))
        if scale < 1.0:
            img = ndimage_zoom(img.astype(np.float32), scale, order=1)
        pd = compute_persistence_diagram(img, max_dim, threshold)
        pds.append(pd)

    sim_matrix = np.zeros((B, B), dtype=np.float32)
    for i in range(B):
        for j in range(i, B):
            s = sim_pd(pds[i], pds[j])
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s

    return sim_matrix
