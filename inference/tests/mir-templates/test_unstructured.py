# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import mir
import numpy as np
import pytest
from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid


class MockTemplateManager:
    pass


@pytest.fixture
def provider() -> MirTemplatesProvider:
    return MirTemplatesProvider(MockTemplateManager())


@pytest.fixture
def state() -> dict:
    """A small cutout-like set of points."""
    rng = np.random.default_rng(0)
    return {
        "latitudes": np.sort(rng.uniform(40.0, 60.0, 300)),
        "longitudes": rng.uniform(-10.0, 10.0, 300),
    }


def _expected_uid(state: dict) -> str:
    return mir.Grid(latitudes=list(state["latitudes"]), longitudes=list(state["longitudes"])).uid


def _check_unstructured(template, state: dict) -> None:
    assert template.metadata("gridType") == "unstructured_grid"
    assert template.metadata("numberOfDataPoints") == len(state["latitudes"])
    assert template.metadata("uuidOfHGrid") == _expected_uid(state)


def test_no_grid_uses_state(provider: MirTemplatesProvider, state: dict):
    template = provider.template("lsm", {"levtype": "sfc", "grid": None}, state=state)
    _check_unstructured(template, state)


def test_missing_grid_key_uses_state(provider: MirTemplatesProvider, state: dict):
    template = provider.template("lsm", {"levtype": "sfc"}, state=state)
    _check_unstructured(template, state)


def test_no_grid_and_no_state(provider: MirTemplatesProvider):
    assert provider.template("lsm", {"levtype": "sfc", "grid": None}) is None
    assert provider.template("lsm", {"levtype": "sfc", "grid": None}, state={}) is None
    assert provider.template("lsm", {"levtype": "sfc", "grid": None}, state={"latitudes": [1.0]}) is None


@pytest.mark.parametrize("levtype", ["sfc", "pl"])
def test_mismatched_grid_uses_state(provider: MirTemplatesProvider, state: dict, levtype: str):
    template = provider.template("lsm", {"levtype": levtype, "grid": "O96"}, state=state)
    _check_unstructured(template, state)


def test_mismatched_grid_and_area_uses_state(provider: MirTemplatesProvider, state: dict):
    """A cutout checkpoint: the lookup has the LAM source's grid and area."""
    template = provider.template("lsm", {"levtype": "sfc", "grid": "0.5/0.5", "area": "10/0/0/10"}, state=state)
    _check_unstructured(template, state)


def test_unbuildable_grid_uses_state(provider: MirTemplatesProvider, state: dict):
    template = provider.template("lsm", {"levtype": "sfc", "grid": "not-a-grid"}, state=state)
    _check_unstructured(template, state)


def test_unbuildable_grid_without_state_raises(provider: MirTemplatesProvider):
    with pytest.raises(Exception):
        provider.template("lsm", {"levtype": "sfc", "grid": "not-a-grid"})


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


def test_coordinates_round_trip(provider: MirTemplatesProvider, state: dict):
    template = provider.template("lsm", {"levtype": "sfc", "grid": None}, state=state)
    latlon = template.to_latlon(flatten=True)
    np.testing.assert_allclose(latlon["lat"], state["latitudes"], atol=1e-4)
    np.testing.assert_allclose(np.mod(latlon["lon"], 360.0), np.mod(state["longitudes"], 360.0), atol=1e-4)


def test_unstructured_regrid_error_propagates(
    provider: MirTemplatesProvider, state: dict, monkeypatch: pytest.MonkeyPatch
):
    forward = MIRRegrid.forward

    def failing_forward(self, fields):
        if isinstance(self.grid, dict):
            raise RuntimeError("MIR failed on the unstructured grid")
        return forward(self, fields)

    monkeypatch.setattr(MIRRegrid, "forward", failing_forward)

    with pytest.raises(RuntimeError, match="unstructured grid"):
        provider.template("lsm", {"levtype": "sfc", "grid": "O96"}, state=state)


def test_wrong_point_count_after_rebuild_raises(
    provider: MirTemplatesProvider, state: dict, monkeypatch: pytest.MonkeyPatch
):
    forward = MIRRegrid.forward

    def other_grid_forward(self, fields):
        if isinstance(self.grid, dict):
            return forward(MIRRegrid(grid="O48"), fields)
        return forward(self, fields)

    monkeypatch.setattr(MIRRegrid, "forward", other_grid_forward)

    with pytest.raises(ValueError, match="points"):
        provider.template("lsm", {"levtype": "sfc", "grid": "O96"}, state=state)


@pytest.mark.parametrize(
    "lookup",
    [
        {"levtype": "sfc", "grid": None},
        {"levtype": "sfc", "grid": "O96"},
        {"levtype": "sfc", "grid": "0.5/0.5", "area": "10/0/0/10"},
        {"levtype": "sfc", "grid": "not-a-grid"},
    ],
)
def test_mismatch_regrids_once(
    provider: MirTemplatesProvider, state: dict, monkeypatch: pytest.MonkeyPatch, lookup: dict
):
    forward = MIRRegrid.forward
    grids = []

    def counting_forward(self, fields):
        grids.append(self.grid)
        return forward(self, fields)

    monkeypatch.setattr(MIRRegrid, "forward", counting_forward)

    template = provider.template("lsm", lookup, state=state)

    _check_unstructured(template, state)
    assert len(grids) == 1
    assert isinstance(grids[0], dict)


def test_matching_grid_regrids_once(provider: MirTemplatesProvider, monkeypatch: pytest.MonkeyPatch):
    npoints = mir.Grid(grid="O96").size()
    state = {"latitudes": np.zeros(npoints), "longitudes": np.zeros(npoints)}
    forward = MIRRegrid.forward
    grids = []

    def counting_forward(self, fields):
        grids.append(self.grid)
        return forward(self, fields)

    monkeypatch.setattr(MIRRegrid, "forward", counting_forward)

    provider.template("lsm", {"levtype": "sfc", "grid": "O96"}, state=state)

    assert grids == ["O96"]


def test_regrid_error_propagates_without_state(provider: MirTemplatesProvider, monkeypatch: pytest.MonkeyPatch):
    def failing_forward(self, fields):
        raise RuntimeError("MIR failed")

    monkeypatch.setattr(MIRRegrid, "forward", failing_forward)

    with pytest.raises(RuntimeError, match="MIR failed"):
        provider.template("lsm", {"levtype": "sfc", "grid": "O96"})


def _count_forwards(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record the grid of every MIRRegrid.forward call."""
    forward = MIRRegrid.forward
    grids = []

    def counting_forward(self, fields):
        grids.append(self.grid)
        return forward(self, fields)

    monkeypatch.setattr(MIRRegrid, "forward", counting_forward)
    return grids


# A coordinate grid, what a named grid such as "meps" resolves to
COORDINATE_GRID = {"latitudes": [50.0, 51.0, 52.0], "longitudes": [0.0, 1.0, 2.0]}


@pytest.mark.parametrize(
    "grid, area",
    [(COORDINATE_GRID, None), (COORDINATE_GRID, [51, 0, 49, 1]), ("meps", None), ("meps", [60, 10, 59, 11])],
    ids=["coordinates", "coordinates-area", "named", "named-area"],
)
def test_coordinate_lookup_grid_uses_state(provider: MirTemplatesProvider, monkeypatch: pytest.MonkeyPatch, grid, area):
    """A lookup grid given as coordinates is not compared: the state coordinates are used, with one regrid."""
    state = {"latitudes": [10.0, 11.0, 12.0], "longitudes": [20.0, 21.0, 22.0]}
    grids = _count_forwards(monkeypatch)

    template = provider.template("lsm", {"levtype": "sfc", "grid": grid, "area": area}, state=state)

    _check_unstructured(template, state)
    assert grids == [state]


def test_coordinate_grid_with_area_regrid_error_without_state_raises(provider: MirTemplatesProvider):
    with pytest.raises(Exception):
        provider.template("lsm", {"levtype": "sfc", "grid": {"latitudes": [1.0], "longitudes": []}, "area": "1/0/0/1"})


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
    monkeypatch: pytest.MonkeyPatch,
    lookup: dict,
    mir_grid: dict,
    grid_type: str,
):
    reference = mir.Grid(**mir_grid)
    latitudes, longitudes = reference.to_latlons()
    state = {"latitudes": np.asarray(latitudes), "longitudes": np.asarray(longitudes)}
    grids = _count_forwards(monkeypatch)

    template = provider.template("lsm", {"levtype": "sfc", **lookup}, state=state)

    assert len(grids) == 1
    assert not isinstance(grids[0], dict)
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


def test_two_dimensional_state_without_grid_is_flattened(provider: MirTemplatesProvider):
    rng = np.random.default_rng(1)
    state = {"latitudes": rng.uniform(40, 60, (10, 20)), "longitudes": rng.uniform(-10, 10, (10, 20))}

    template = provider.template("lsm", {"levtype": "sfc", "grid": None}, state=state)

    assert template.metadata("gridType") == "unstructured_grid"
    assert template.metadata("numberOfDataPoints") == 200
