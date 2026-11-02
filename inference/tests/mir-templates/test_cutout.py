# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Templates for stretched-grid (cutout) output.

A cutout joins a regular limited area model (LAM) grid, the first source, with
the points of a global Gaussian grid that lie outside the LAM box, the second
source. The output state is the LAM points followed by those global points.

The template lookup only describes one source at a time, so what a caller
passes depends on how the output is produced:

- no extraction: the global grid and a global area, describing only the global source
- extraction of the LAM source: a bare numeric increment (anemoi-inference
  collapses ``[d, d]`` to ``d``) and the LAM area
- extraction of the global source: the global grid, and the extent of the
  points it has left

Whenever the lookup does not describe the points of the output, the template
has to be an unstructured grid built from the state's coordinates.

The global coordinates are derived with MIR.
"""

from dataclasses import dataclass

import mir
import numpy as np
import pytest
from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider


@dataclass(frozen=True)
class Layout:
    """A cutout of a regular LAM into a global Gaussian grid."""

    grid: str  # global grid, e.g. "O32"
    increment: float  # LAM grid increment, in degrees
    area: tuple[float, float, float, float]  # LAM north, west, south, east

    @property
    def lam_area(self) -> list[float]:
        """The LAM area as a lookup would have it, longitudes in [0, 360]."""
        north, west, south, east = self.area
        return [north, west % 360.0, south, east % 360.0]

    @property
    def lam_grid(self) -> str:
        return f"{self.increment}/{self.increment}"


LAYOUTS = {
    "mid_latitude": Layout("O32", 1.0, (60.0, 0.0, 30.0, 40.0)),
    "half_degree": Layout("O48", 0.5, (55.0, 5.0, 35.0, 25.0)),
    "crosses_meridian": Layout("O32", 1.0, (60.0, -15.0, 40.0, 15.0)),
    "crosses_dateline": Layout("O32", 1.0, (50.0, 170.0, 30.0, 190.0)),
}


class MockTemplateManager:
    pass


@pytest.fixture(scope="module")
def provider() -> MirTemplatesProvider:
    return MirTemplatesProvider(MockTemplateManager())


def build_states(provider: MirTemplatesProvider, layout: Layout) -> dict[str, dict]:
    """The states of the full cutout, and of its LAM and global parts."""
    north, west, south, east = layout.area
    d = layout.increment

    nlat = round((north - south) / d) + 1
    nlon = round((east - west) / d) + 1
    lam_lat, lam_lon = np.meshgrid(north - d * np.arange(nlat), west + d * np.arange(nlon), indexing="ij")
    lam_lat, lam_lon = lam_lat.ravel(), lam_lon.ravel()

    globe = provider.template("lsm", {"levtype": "sfc", "grid": layout.grid}).to_latlon(flatten=True)
    glob_lat, glob_lon = globe["lat"], globe["lon"]
    tolerance = 1e-6
    inside = (
        (glob_lat <= north + tolerance)
        & (glob_lat >= south - tolerance)
        & (np.mod(glob_lon - west, 360.0) <= (east - west) + tolerance)
    )

    lam = {"latitudes": lam_lat, "longitudes": lam_lon}
    glob = {"latitudes": glob_lat[~inside], "longitudes": glob_lon[~inside]}
    full = {key: np.concatenate([lam[key], glob[key]]) for key in lam}

    # The box must cut points out of the global grid, and leave some in it
    assert 0 < inside.sum() < len(glob_lat)
    assert len(full["latitudes"]) == len(lam["latitudes"]) + len(glob["latitudes"])
    return {"full": full, "lam": lam, "global": glob}


@pytest.fixture(scope="module", params=list(LAYOUTS))
def layout(request: pytest.FixtureRequest) -> Layout:
    return LAYOUTS[request.param]


@pytest.fixture(scope="module")
def states(provider: MirTemplatesProvider, layout: Layout) -> dict[str, dict]:
    return build_states(provider, layout)


def extent_area(state: dict) -> list[float]:
    """The extent of a set of points, as an area with longitudes in [0, 360]."""
    latitudes = state["latitudes"]
    longitudes = np.mod(state["longitudes"], 360.0)
    return [float(latitudes.max()), float(longitudes.min()), float(latitudes.min()), float(longitudes.max())]


def _uid(state: dict) -> str:
    return mir.Grid(**{k: list(v) for k, v in state.items()}).uid


def _assert_coordinates(template, state: dict) -> None:
    latlon = template.to_latlon(flatten=True)
    np.testing.assert_allclose(latlon["lat"], state["latitudes"], atol=1e-4)
    np.testing.assert_allclose(np.mod(latlon["lon"], 360.0), np.mod(state["longitudes"], 360.0), atol=1e-4)


def _assert_unstructured(template, state: dict) -> None:
    assert template.metadata("gridType") == "unstructured_grid"
    assert template.metadata("numberOfDataPoints") == len(state["latitudes"])
    assert template.metadata("uuidOfHGrid") == _uid(state)
    _assert_coordinates(template, state)


def _assert_regular(template, state: dict) -> None:
    assert template.metadata("gridType") == "regular_ll"
    assert template.metadata("numberOfDataPoints") == len(state["latitudes"])
    _assert_coordinates(template, state)


@pytest.mark.parametrize("variable, levtype", [("2t", "sfc"), ("t", "pl")])
def test_full_output_with_global_grid(
    provider: MirTemplatesProvider, layout: Layout, states: dict, variable: str, levtype: str
):
    """The lookup describes the global source only, the output has the LAM points too."""
    state = states["full"]
    lookup = {"levtype": levtype, "grid": layout.grid, "area": [90, 0.0, -90, 360]}

    template = provider.template(variable, lookup, state=state)

    _assert_unstructured(template, state)


def test_full_output_with_global_grid_only(provider: MirTemplatesProvider, layout: Layout, states: dict):
    state = states["full"]

    template = provider.template("2t", {"levtype": "sfc", "grid": layout.grid}, state=state)

    _assert_unstructured(template, state)


def test_full_output_with_lam_grid(provider: MirTemplatesProvider, layout: Layout, states: dict):
    """The lookup describes the LAM source only, the output has the global points too."""
    state = states["full"]
    lookup = {"levtype": "sfc", "grid": layout.increment, "area": layout.lam_area}

    template = provider.template("2t", lookup, state=state)

    _assert_unstructured(template, state)


@pytest.mark.parametrize("bare", [True, False], ids=["bare_increment", "explicit_grid"])
def test_lam_part_is_regular(provider: MirTemplatesProvider, layout: Layout, states: dict, bare: bool):
    """A bare increment (or ``d/d``) and the LAM area are enough for a regular lat/lon grid."""
    state = states["lam"]
    grid = layout.increment if bare else layout.lam_grid

    template = provider.template("2t", {"levtype": "sfc", "grid": grid, "area": layout.lam_area}, state=state)

    _assert_regular(template, state)


@pytest.mark.parametrize("with_area", [False, True], ids=["no_area", "extent_area"])
def test_global_part_is_unstructured(provider: MirTemplatesProvider, layout: Layout, states: dict, with_area: bool):
    """The global points outside the LAM are not a Gaussian grid, the template follows the state."""
    state = states["global"]
    lookup = {"levtype": "sfc", "grid": layout.grid}
    if with_area:
        lookup["area"] = extent_area(state)

    template = provider.template("2t", lookup, state=state)

    _assert_unstructured(template, state)


def test_unstructured_templates_differ_by_points(provider: MirTemplatesProvider, layout: Layout, states: dict):
    full = provider.template("2t", {"levtype": "sfc", "grid": layout.grid}, state=states["full"])
    glob = provider.template("2t", {"levtype": "sfc", "grid": layout.grid}, state=states["global"])

    assert full.metadata("numberOfDataPoints") != glob.metadata("numberOfDataPoints")
    assert full.metadata("uuidOfHGrid") != glob.metadata("uuidOfHGrid")


def test_layouts_have_different_uids(provider: MirTemplatesProvider):
    uids = {}
    for name, layout in LAYOUTS.items():
        state = build_states(provider, layout)["full"]
        template = provider.template("2t", {"levtype": "sfc", "grid": layout.grid}, state=state)
        uids[name] = template.metadata("uuidOfHGrid")

    assert len(set(uids.values())) == len(LAYOUTS)
