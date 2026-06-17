"""Config-driven canyon geometry: dimensions, masks, aspect-ratio handling."""

import numpy as np
import pytest

from canyon_lbm.geometry import build_canyon


def test_dimensions_follow_config():
    g = build_canyon(cells_per_H=40, aspect_ratio=1.0, building_width_H=1.0,
                     fetch_upstream_H=5.0, top_margin_H=6.0, outflow_H=15.0)
    assert g.h == 40 and g.w == 40 and g.b == 40
    # nx = fetch + building + street + building + outflow (cells)
    assert g.nx == 200 + 40 + 40 + 40 + 600
    # ny = ground row + building height + top margin
    assert g.ny == 1 + 40 + 240


def test_solid_mask_structure():
    g = build_canyon(cells_per_H=32, aspect_ratio=1.0)
    # Ground is solid across the whole bottom row; row 1 in the street is fluid.
    assert g.solid[0].all()
    s0, s1 = g.street
    assert not g.solid[1, s0:s1].any()
    # Buildings are solid from row 1 to the roof, fluid just above the roof.
    w0, w1 = g.west_building
    assert g.solid[1 : g.h + 1, w0:w1].all()
    assert not g.solid[g.h + 1, w0:w1].any()
    assert g.roof_row == g.h


def test_cavity_mask_counts_street_cells():
    g = build_canyon(cells_per_H=24, aspect_ratio=1.5)
    assert g.cavity_mask.sum() == g.h * g.w
    # Cavity cells are all fluid and all within the street columns.
    assert not (g.cavity_mask & g.solid).any()
    s0, s1 = g.street
    cols = np.where(g.cavity_mask.any(axis=0))[0]
    assert cols.min() == s0 and cols.max() == s1 - 1


@pytest.mark.parametrize("ar,expected_w", [(0.5, 64), (1.0, 32), (2.0, 16)])
def test_aspect_ratio_sets_street_width(ar, expected_w):
    g = build_canyon(cells_per_H=32, aspect_ratio=ar)
    assert g.w == expected_w
    assert g.aspect_ratio_actual == pytest.approx(g.h / g.w)


def test_building_blocks_are_contiguous_and_ordered():
    g = build_canyon(cells_per_H=20, aspect_ratio=1.0)
    assert g.west_building[1] == g.street[0]
    assert g.street[1] == g.east_building[0]
    assert g.west_building[0] == g.L_up
