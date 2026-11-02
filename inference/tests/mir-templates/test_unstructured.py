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
