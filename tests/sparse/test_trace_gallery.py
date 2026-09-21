"""Spatial coordinates must be derived from the real model grid and preprocessing."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sparse"))
from trace_gallery import token_mapping


def test_real_grid_boundaries_cameras_and_future_do_not_claim_observed_rgb():
    mapping = token_mapping((3, 7, 14), (224, 448, 3),
                            {"agentview": (256, 256, 3), "wrist": (256, 320, 3)})
    assert len(mapping) == 294
    assert mapping[0]["model_input_bbox"] == [0, 0, 32, 32]
    assert mapping[6]["camera"] == "agentview"
    assert mapping[7]["camera"] == "wrist"
    assert mapping[7]["raw_observation_bbox"][0] == 32
    assert mapping[98]["latent_frame"] == 1
    assert mapping[98]["raw_observation_bbox"] is None
    assert not mapping[293]["observed_rgb"]
    assert mapping[293]["row"] == 6 and mapping[293]["col"] == 13


def test_grid_cannot_silently_cross_camera_seam():
    with pytest.raises(ValueError, match="aligned"):
        token_mapping((3, 7, 13), (224, 448, 3), {})
