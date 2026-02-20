"""
Tests for engine/inference.py — pure numpy, zero GPU or MuseTalk dependencies.

compute_feathered_mask is importable without musetalk because all musetalk
imports in inference.py are lazy (inside function bodies).
"""

import numpy as np
import pytest


def test_mask_shape():
    """Output shape matches the requested (h, w) with dtype float32."""
    from engine.inference import compute_feathered_mask

    mask = compute_feathered_mask(100, 80, feather=10)
    assert mask.shape == (100, 80)
    assert mask.dtype == np.float32


def test_mask_corners_less_than_centre():
    """Corner pixel value is less than the centre pixel value."""
    from engine.inference import compute_feathered_mask

    mask = compute_feathered_mask(100, 80, feather=10)
    assert mask[0, 0] < mask[50, 40]


def test_mask_centre_is_one():
    """The centre pixel value is exactly 1.0 (unaffected by feathering)."""
    from engine.inference import compute_feathered_mask

    mask = compute_feathered_mask(100, 80, feather=10)
    assert mask[50, 40] == 1.0


def test_mask_values_in_range():
    """All mask values are within [0.0, 1.0]."""
    from engine.inference import compute_feathered_mask

    mask = compute_feathered_mask(100, 80, feather=10)
    assert np.all(mask >= 0.0)
    assert np.all(mask <= 1.0)


def test_mask_symmetric():
    """Top row equals bottom row; left column equals right column."""
    from engine.inference import compute_feathered_mask

    mask = compute_feathered_mask(100, 80, feather=10)
    np.testing.assert_allclose(mask[0, :], mask[-1, :])
    np.testing.assert_allclose(mask[:, 0], mask[:, -1])
