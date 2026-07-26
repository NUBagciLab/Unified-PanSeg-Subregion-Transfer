"""Segmentation post-processing on numpy/torch arrays (no extra I/O)."""

from typing import Union

import numpy as np
import torch
from scipy import ndimage


def largest_connected_component_per_class(
    segmentation: Union[np.ndarray, torch.Tensor],
    background_label: int = 0,
) -> np.ndarray:
    """
    For each non-background label, keep only the largest 3D connected component.

    Args:
        segmentation: Label map (H, W, D) with integer class ids.
        background_label: Label value to leave unchanged (typically 0).

    Returns:
        New label map of the same shape and dtype; input is not modified.
    """
    if isinstance(segmentation, torch.Tensor):
        seg = segmentation.detach().cpu().numpy()
    else:
        seg = np.asarray(segmentation)

    out = np.zeros_like(seg, dtype=seg.dtype)
    for label_id in np.unique(seg):
        if label_id == background_label:
            continue
        mask = seg == label_id
        if not mask.any():
            continue
        labeled, num_components = ndimage.label(mask)
        if num_components == 0:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
        largest_id = int(np.argmax(sizes)) + 1
        out[labeled == largest_id] = label_id
    return out
