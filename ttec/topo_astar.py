"""
Gradient-Free Topo-A* Correction.
Only activated for Type-III errors. Operates on probability maps, no weight updates.
"""
import numpy as np
import heapq
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import label as scipy_label, binary_dilation, distance_transform_edt

from data.topology import compute_betti_numbers


class TopoAStar:
    """A* search with topological cost function for mask correction."""

    def __init__(self, config):
        self.tau_anchor_vac = config.tau_anchor_vacuity
        self.anchor_conf = config.anchor_confidence
        self.lambda_topo = config.lambda_topo
        self.d0 = config.repulsion_cutoff
        self.sigma_path = config.path_sigma
        self.rho = config.path_rho
        self.max_iters = config.max_iterations

    def correct(self, prediction: np.ndarray, prob_maps: Dict[str, np.ndarray],
                vacuity: np.ndarray, violations: Dict,
                betti_targets: Dict) -> np.ndarray:
        """Apply structure-specific corrections for Type-III violations.

        Args:
            prediction: (H, W) integer segmentation
            prob_maps: {structure: (H, W) probability}
            vacuity: (H, W) vacuity map
            violations: from TTEC, only Type-III entries
            betti_targets: target Betti numbers per structure

        Returns:
            Corrected prediction (H, W)
        """
        corrected = prediction.copy()

        for structure, viol in violations.items():
            if viol["type"].value != "genuine_model_error":
                continue

            actual = viol["actual_betti"]
            target = betti_targets.get(structure)
            if target is None:
                continue

            class_idx = {"LV": 1, "Myo": 2, "RV": 3}[structure]
            mask = (corrected == class_idx).astype(np.uint8)
            prob = prob_maps.get(structure, np.zeros_like(vacuity))

            if structure == "Myo":
                new_mask = self._correct_myo(mask, prob, vacuity, target)
            elif structure == "LV":
                new_mask = self._correct_lv(mask, prob, vacuity, corrected)
            elif structure == "RV":
                new_mask = self._correct_rv(mask, prob, vacuity)
            else:
                continue

            # Verify correction
            new_betti = compute_betti_numbers(new_mask)
            if new_betti == target:
                # Apply correction
                corrected[mask == 1] = 0  # Clear old
                corrected[new_mask == 1] = class_idx
            # Else: discard correction, keep original

        return corrected

    # ---- Anchors ----

    def _get_anchors(self, prob: np.ndarray, vacuity: np.ndarray) -> np.ndarray:
        """Evidential anchors: low vacuity + high confidence."""
        return ((vacuity < self.tau_anchor_vac) & (prob > self.anchor_conf)).astype(np.uint8)

    def _get_background_pixels(self, prediction: np.ndarray, class_idx: int) -> np.ndarray:
        """High-confidence background pixels for repulsive field."""
        return (prediction == 0).astype(np.uint8)

    # ---- Potential Fields ----

    def _attractive_field(self, point: Tuple[int, int],
                          anchors: np.ndarray) -> float:
        anchor_coords = np.argwhere(anchors > 0)
        if len(anchor_coords) == 0:
            return 0.0
        dists = np.sum((anchor_coords - np.array(point)) ** 2, axis=1) + 1e-6
        return -np.sum(1.0 / dists)

    def _repulsive_field(self, point: Tuple[int, int],
                         bg_pixels: np.ndarray) -> float:
        bg_coords = np.argwhere(bg_pixels > 0)
        if len(bg_coords) == 0:
            return 0.0
        dists_sq = np.sum((bg_coords - np.array(point)) ** 2, axis=1)
        close = dists_sq < self.d0 ** 2
        if not close.any():
            return 0.0
        return np.sum(np.maximum(0, 1.0 / (dists_sq[close] + 1e-6) - 1.0 / self.d0 ** 2))

    # ---- A* Search ----

    def _astar_bridge(self, mask: np.ndarray, prob: np.ndarray,
                      vacuity: np.ndarray, start: Tuple, goal: Tuple,
                      betti_target: Tuple) -> List[Tuple]:
        """A* search with topological cost: f(n) = g(n) + lambda*delta_beta + h(n)."""
        H, W = mask.shape
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        visited = set()

        for iteration in range(self.max_iters):
            if not open_set:
                break

            _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if current in visited:
                continue
            visited.add(current)

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < H and 0 <= nx < W:
                    neighbor = (ny, nx)
                    if neighbor in visited:
                        continue

                    # Path cost (Euclidean)
                    step_cost = np.sqrt(dy ** 2 + dx ** 2)
                    # Prefer high-probability pixels
                    prob_cost = 1.0 - prob[ny, nx] + 0.01
                    tentative_g = g_score[current] + step_cost * prob_cost

                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g

                        # Heuristic: Euclidean distance to goal
                        h = np.sqrt((ny - goal[0]) ** 2 + (nx - goal[1]) ** 2)

                        # Topological cost: delta_beta
                        delta_beta = self._estimate_delta_beta(mask, neighbor, betti_target)

                        f = tentative_g + self.lambda_topo * delta_beta + h
                        heapq.heappush(open_set, (f, neighbor))

        return []  # No path found

    def _estimate_delta_beta(self, mask, pixel, target_betti):
        """Quick estimate of Betti change if pixel is added to mask."""
        # Approximate: count how many 4-connected neighbors are in mask
        y, x = pixel
        H, W = mask.shape
        neighbors_in = 0
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx]:
                neighbors_in += 1

        if neighbors_in == 0:
            return 1.0  # Isolated pixel → increases beta0
        elif neighbors_in >= 2:
            return 0.0  # Bridges → may fix topology
        return 0.5

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    # ---- Structure-Specific Corrections ----

    def _correct_myo(self, mask, prob, vacuity, target):
        """Bridge ruptured Myo ring using A* search."""
        anchors = self._get_anchors(prob, vacuity)
        labeled, n_components = scipy_label(mask)

        if n_components <= 1:
            return mask  # Already connected or empty

        # Find two largest components to bridge
        sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
        sorted_idx = np.argsort(sizes)[::-1]

        if len(sorted_idx) < 2:
            return mask

        c1_mask = (labeled == sorted_idx[0] + 1)
        c2_mask = (labeled == sorted_idx[1] + 1)

        # Find closest pair of boundary pixels
        from scipy.ndimage import binary_erosion
        b1 = c1_mask & ~binary_erosion(c1_mask)
        b2 = c2_mask & ~binary_erosion(c2_mask)

        pts1 = np.argwhere(b1)
        pts2 = np.argwhere(b2)

        if len(pts1) == 0 or len(pts2) == 0:
            return mask

        from scipy.spatial.distance import cdist
        dists = cdist(pts1, pts2)
        i, j = np.unravel_index(dists.argmin(), dists.shape)
        start = tuple(pts1[i])
        goal = tuple(pts2[j])

        # A* bridge
        path = self._astar_bridge(mask, prob, vacuity, start, goal, target)

        if path:
            new_mask = mask.copy()
            for y, x in path:
                # Gaussian-weighted integration
                yy, xx = np.ogrid[max(0, y-3):min(mask.shape[0], y+4),
                                  max(0, x-3):min(mask.shape[1], x+4)]
                dist_sq = (yy - y)**2 + (xx - x)**2
                weight = np.exp(-dist_sq / (2 * self.sigma_path**2))
                region = new_mask[max(0, y-3):min(mask.shape[0], y+4),
                                 max(0, x-3):min(mask.shape[1], x+4)]
                new_mask[max(0, y-3):min(mask.shape[0], y+4),
                         max(0, x-3):min(mask.shape[1], x+4)] = np.maximum(
                    region, (weight > 0.3).astype(np.uint8))
            return new_mask

        return mask

    def _correct_lv(self, mask, prob, vacuity, full_pred):
        """Fill LV holes using morphological flood fill bounded by Myo."""
        from scipy.ndimage import binary_fill_holes
        myo_mask = (full_pred == 2)
        filled = binary_fill_holes(mask | myo_mask).astype(np.uint8)
        # LV = filled region minus Myo
        lv_filled = filled & ~myo_mask
        return lv_filled.astype(np.uint8)

    def _correct_rv(self, mask, prob, vacuity):
        """Reconnect RV fragments."""
        labeled, n = scipy_label(mask)
        if n <= 1:
            return mask

        # Keep largest component, try to connect others via dilation
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        largest = np.argmax(sizes) + 1
        new_mask = (labeled == largest).astype(np.uint8)

        # Iterative dilation to connect nearby fragments
        for _ in range(5):
            dilated = binary_dilation(new_mask, iterations=1)
            # Check if any other component is now connected
            for c in range(1, n + 1):
                if c == largest:
                    continue
                if (dilated & (labeled == c)).any():
                    new_mask = new_mask | (labeled == c).astype(np.uint8)

        return new_mask
