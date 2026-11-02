# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Templates for a stretched-grid cutout, laid out like BRIS-Europe-2025.10.

That model is a 0.5/0.5 LAM over 70/-20/20/30 cut into an O48 global grid.
Its 20657 points are the LAM points (101 x 101 = 10201, `source0`) followed
by 10456 of the 10944 O48 points (`source1`). The real cutout drops a few more
O48 points around the LAM than this test does, so only the structure is pinned.

The lookup values below are what anemoi-inference passes for that checkpoint:

- no extraction: the metadata grid and area (O48, global), which only describe `source1`
- `extract_from_state: source0`: the state's ``_geography`` (a bare ``0.5``, as
  anemoi-inference collapses ``[0.5, 0.5]``, and the LAM area)
- `extract_from_state: source1`: the state's ``_geography`` (O48 and the extent
  of its remaining points)

No checkpoint is needed, the coordinates are derived with MIR.
"""

import mir
import numpy as np
import pytest
from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider

# From BRIS-Europe-2025.10
LAM_AREA = (70.0, -20.0, 20.0, 30.0)  # north, west, south, east
N_LAM = 10201
N_O48 = 10944
LOOKUPS = {
    "full": {"grid": "O48", "area": [90, 0.0, -90, 360]},
    "lam": {"grid": 0.5, "area": [70.0, 340.0, 20.0, 30.0]},
    "global": {"grid": "O48", "area": [88.572169, 0.0, -88.572169, 358.269231]},
}


class MockTemplateManager:
    pass


@pytest.fixture(scope="module")
def provider() -> MirTemplatesProvider:
    return MirTemplatesProvider(MockTemplateManager())


@pytest.fixture(scope="module")
def states(provider: MirTemplatesProvider) -> dict[str, dict]:
    """The states of the full cutout, and of its LAM and global parts."""
    north, west, south, east = LAM_AREA
    lam_lat, lam_lon = np.meshgrid(
        np.arange(north, south - 0.25, -0.5), np.arange(west, east + 0.25, 0.5), indexing="ij"
    )
    lam_lat, lam_lon = lam_lat.ravel(), lam_lon.ravel()

    o48 = provider.template("lsm", {"levtype": "sfc", "grid": "O48"}).to_latlon(flatten=True)
    glob_lat, glob_lon = o48["lat"], o48["lon"]
    wrapped = (glob_lon + 180.0) % 360.0 - 180.0
    outside = ~((glob_lat <= north) & (glob_lat >= south) & (wrapped >= west) & (wrapped <= east))

    lam = {"latitudes": lam_lat, "longitudes": lam_lon}
    glob = {"latitudes": glob_lat[outside], "longitudes": glob_lon[outside]}
    full = {key: np.concatenate([lam[key], glob[key]]) for key in lam}

    assert len(lam["latitudes"]) == N_LAM
    assert N_O48 - 1000 < len(glob["latitudes"]) < N_O48
    assert len(full["latitudes"]) == N_LAM + len(glob["latitudes"])
    return {"full": full, "lam": lam, "global": glob}


def _assert_coordinates(template, state: dict) -> None:
    latlon = template.to_latlon(flatten=True)
    np.testing.assert_allclose(latlon["lat"], state["latitudes"], atol=1e-4)
    np.testing.assert_allclose(np.mod(latlon["lon"], 360.0), np.mod(state["longitudes"], 360.0), atol=1e-4)


@pytest.mark.parametrize("variable, levtype", [("2t", "sfc"), ("t", "pl")])
def test_full_cutout_is_unstructured(provider: MirTemplatesProvider, states: dict, variable: str, levtype: str):
    """The metadata grid is the O48 source only (10944 points), the output has all the cutout points."""
    state = states["full"]

    template = provider.template(variable, {"levtype": levtype, **LOOKUPS["full"]}, state=state)

    assert template.metadata("gridType") == "unstructured_grid"
    assert template.metadata("numberOfDataPoints") == len(state["latitudes"])
    assert template.metadata("uuidOfHGrid") == mir.Grid(**{k: list(v) for k, v in state.items()}).uid
    _assert_coordinates(template, state)


def test_lam_extract_is_regular(provider: MirTemplatesProvider, states: dict):
    """A bare 0.5 grid and the LAM area are enough: the template is a regular lat/lon grid."""
    state = states["lam"]

    template = provider.template("2t", {"levtype": "sfc", **LOOKUPS["lam"]}, state=state)

    assert template.metadata("gridType") == "regular_ll"
    assert template.metadata("numberOfDataPoints") == N_LAM
    _assert_coordinates(template, state)


def test_lam_extract_with_explicit_grid(provider: MirTemplatesProvider, states: dict):
    state = states["lam"]

    template = provider.template(
        "2t", {"levtype": "sfc", "grid": "0.5/0.5", "area": LOOKUPS["lam"]["area"]}, state=state
    )

    assert template.metadata("gridType") == "regular_ll"
    assert template.metadata("numberOfDataPoints") == N_LAM
    _assert_coordinates(template, state)


def test_global_extract_is_unstructured(provider: MirTemplatesProvider, states: dict):
    """The O48 points outside the LAM are not an O48 grid, the template follows the state."""
    state = states["global"]

    template = provider.template("2t", {"levtype": "sfc", **LOOKUPS["global"]}, state=state)

    assert template.metadata("gridType") == "unstructured_grid"
    assert template.metadata("numberOfDataPoints") == len(state["latitudes"])
    assert template.metadata("uuidOfHGrid") == mir.Grid(**{k: list(v) for k, v in state.items()}).uid
    _assert_coordinates(template, state)


def test_unstructured_templates_differ_by_points(provider: MirTemplatesProvider, states: dict):
    full = provider.template("2t", {"levtype": "sfc", **LOOKUPS["full"]}, state=states["full"])
    glob = provider.template("2t", {"levtype": "sfc", **LOOKUPS["global"]}, state=states["global"])

    assert full.metadata("uuidOfHGrid") != glob.metadata("uuidOfHGrid")
