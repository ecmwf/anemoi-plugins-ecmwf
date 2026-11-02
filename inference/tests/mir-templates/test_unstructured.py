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
    assert template.handle.get("gridType") == "unstructured_grid"
    assert template.handle.get("numberOfDataPoints") == len(state["latitudes"])
    assert template.handle.get("uuidOfHGrid") == _expected_uid(state)


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


def test_unbuildable_grid_uses_state(provider: MirTemplatesProvider, state: dict):
    template = provider.template("lsm", {"levtype": "sfc", "grid": "not-a-grid"}, state=state)
    _check_unstructured(template, state)


def test_matching_grid_is_unchanged(provider: MirTemplatesProvider):
    grid = mir.Grid(grid="O96")
    npoints = grid.size()
    assert npoints == 40320
    state = {"latitudes": np.zeros(npoints), "longitudes": np.zeros(npoints)}

    template = provider.template("lsm", {"levtype": "sfc", "grid": "O96"}, state=state)

    assert template.handle.get("gridType") == "reduced_gg"
    assert template.metadata().geography.mars_grid() == "O96"
    assert template.handle.get("numberOfDataPoints") == npoints


def test_grid_with_area_ignores_state(provider: MirTemplatesProvider, state: dict):
    template = provider.template("lsm", {"levtype": "sfc", "grid": "0.5/0.5", "area": "10/-0.5/0.5/11"}, state=state)
    assert template.handle.get("gridType") == "regular_ll"
    assert template.metadata().geography.mars_area() == [10.0, 359.5, 0.5, 11.0]
