# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Templates for output states on structured grids: the template keeps that grid."""

import mir
import numpy as np
import pytest
from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid


def test_matching_grid_is_unchanged(provider: MirTemplatesProvider):
    npoints = mir.Grid(grid="O96").size()
    assert npoints == 40320
    state = {"latitudes": np.zeros(npoints), "longitudes": np.zeros(npoints)}

    template = provider.template("lsm", {"levtype": "sfc", "grid": "O96"}, state=state)

    assert template.metadata("gridType") == "reduced_gg"
    assert template.metadata().geography.mars_grid() == "O96"
    assert template.metadata("numberOfDataPoints") == npoints


def test_matching_grid_and_area_is_unchanged(provider: MirTemplatesProvider):
    lookup = {"levtype": "sfc", "grid": "0.5/0.5", "area": "10/0/0/10"}
    expected = provider.template("lsm", lookup)
    latitudes, longitudes = expected.to_latlon(flatten=True).values()
    state = {"latitudes": latitudes, "longitudes": longitudes}

    template = provider.template("lsm", lookup, state=state)

    assert template.metadata("gridType") == "regular_ll"
    assert template.metadata().geography.mars_area() == expected.metadata().geography.mars_area()
    assert template.metadata("numberOfDataPoints") == len(latitudes)


def test_matching_grid_regrids_once(provider: MirTemplatesProvider, forward_grids: list):
    npoints = mir.Grid(grid="O96").size()
    state = {"latitudes": np.zeros(npoints), "longitudes": np.zeros(npoints)}

    provider.template("lsm", {"levtype": "sfc", "grid": "O96"}, state=state)

    assert forward_grids == ["O96"]


def test_structured_grid_regrid_error_with_state_propagates(
    provider: MirTemplatesProvider, monkeypatch: pytest.MonkeyPatch
):
    """The grid matches the state, so a failing regrid is not hidden by a fallback."""
    npoints = mir.Grid(grid="O48").size()
    state = {"latitudes": np.zeros(npoints), "longitudes": np.zeros(npoints)}

    def failing_forward(self, fields):
        raise RuntimeError("MIR failed")

    monkeypatch.setattr(MIRRegrid, "forward", failing_forward)

    with pytest.raises(RuntimeError, match="MIR failed"):
        provider.template("lsm", {"levtype": "sfc", "grid": "O48"}, state=state)


# Structured output states: the coordinates are those of a structured grid, and
# the template must stay that structured grid rather than become unstructured.
STRUCTURED = [
    pytest.param({"grid": "O48"}, {"grid": "O48"}, "reduced_gg", id="O48"),
    pytest.param({"grid": "o48"}, {"grid": "O48"}, "reduced_gg", id="o48-lowercase"),
    pytest.param({"grid": "N320"}, {"grid": "N320"}, "reduced_gg", id="N320"),
    pytest.param({"grid": "n320"}, {"grid": "N320"}, "reduced_gg", id="n320-lowercase"),
    pytest.param({"grid": "1.0/1.0"}, {"grid": "1.0/1.0"}, "regular_ll", id="1.0/1.0"),
    pytest.param({"grid": 1.0}, {"grid": "1.0/1.0"}, "regular_ll", id="1.0-bare"),
    pytest.param({"grid": [1.0, 1.0]}, {"grid": [1.0, 1.0]}, "regular_ll", id="[1.0,1.0]"),
    pytest.param(
        {"grid": "0.5/0.5", "area": "70/-20/20/30"},
        {"grid": "0.5/0.5", "area": "70/-20/20/30"},
        "regular_ll",
        id="0.5/0.5-area",
    ),
    pytest.param(
        {"grid": 0.5, "area": [70, -20, 20, 30]},
        {"grid": "0.5/0.5", "area": [70, -20, 20, 30]},
        "regular_ll",
        id="0.5-bare-area",
    ),
]


@pytest.mark.parametrize(("lookup", "mir_grid", "grid_type"), STRUCTURED)
def test_structured_state_stays_structured(
    provider: MirTemplatesProvider,
    forward_grids: list,
    lookup: dict,
    mir_grid: dict,
    grid_type: str,
):
    reference = mir.Grid(**mir_grid)
    latitudes, longitudes = reference.to_latlons()
    state = {"latitudes": np.asarray(latitudes), "longitudes": np.asarray(longitudes)}

    template = provider.template("lsm", {"levtype": "sfc", **lookup}, state=state)

    assert len(forward_grids) == 1
    assert not isinstance(forward_grids[0], dict)
    assert template.metadata("gridType") == grid_type
    assert template.metadata("numberOfDataPoints") == reference.size()
    assert template.metadata("uuidOfHGrid", default=None) in (None, "00000000000000000000000000000000")

    latlon = template.to_latlon(flatten=True)
    np.testing.assert_allclose(latlon["lat"], state["latitudes"], atol=1e-4)
    np.testing.assert_allclose(np.mod(latlon["lon"], 360.0), np.mod(state["longitudes"], 360.0), atol=1e-4)


@pytest.mark.parametrize(
    "lookup",
    [
        {"grid": "1.0/1.0"},
        {"grid": 1.0},
        {"grid": "0.5/0.5", "area": "70/-20/20/30"},
    ],
)
def test_two_dimensional_state_is_flattened(provider: MirTemplatesProvider, lookup: dict):
    """A regular grid's meshgrid-shaped coordinates behave like flat ones."""
    lookup = {"levtype": "sfc", **lookup}
    flat = provider.template("lsm", lookup)
    latlon = flat.to_latlon(flatten=True)

    lats = np.unique(np.round(latlon["lat"], 6))[::-1]
    lons = np.unique(np.round(latlon["lon"], 6))
    assert lats.size * lons.size == flat.metadata("numberOfDataPoints")
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    assert lat2d.ndim == 2

    template = provider.template("lsm", lookup, state={"latitudes": lat2d, "longitudes": lon2d})

    assert template.metadata("gridType") == "regular_ll"
    assert template.metadata("numberOfDataPoints") == flat.metadata("numberOfDataPoints")
    assert template.metadata().geography.mars_area() == flat.metadata().geography.mars_area()
