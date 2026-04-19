"""
Helpers for building 2.5D context stacks from ordered 2-D slices.
"""
from __future__ import annotations

from typing import Dict, List


def build_context_indices(
    records: List[Dict],
    context_slices: int = 1,
) -> List[List[int]]:
    """
    Return, for each slice index, the neighboring indices to stack as channels.

    Context grouping is done per `(patient_id, phase)` so ED/ES and subjects do
    not bleed into each other. Border slices use edge replication.
    """
    if context_slices <= 1:
        return [[i] for i in range(len(records))]
    if context_slices % 2 == 0:
        raise ValueError(f"context_slices must be odd, got {context_slices}")

    groups: Dict[tuple, List[int]] = {}
    for idx, record in enumerate(records):
        key = (
            record.get("patient_id", ""),
            record.get("phase", ""),
        )
        groups.setdefault(key, []).append(idx)

    context_map: List[List[int]] = [[i] for i in range(len(records))]
    radius = context_slices // 2
    for indices in groups.values():
        ordered = sorted(indices, key=lambda i: int(records[i].get("slice_idx", -1)))
        for pos, center_idx in enumerate(ordered):
            ctx = []
            for offset in range(-radius, radius + 1):
                neighbor_pos = min(max(pos + offset, 0), len(ordered) - 1)
                ctx.append(ordered[neighbor_pos])
            context_map[center_idx] = ctx
    return context_map
