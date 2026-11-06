# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Templates built with real MIR, from the cases of ``cases.py``.

The provider logic (which grid is chosen, what is passed to MIR) is tested
without MIR in ``test_provider.py``. Here MIR builds the templates.
"""

from typing import Any

import pytest
from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider
from cases import CASES
from cases import LAYOUTS
from cases import NULL_UID
from cases import TemplateCase
from cases import assert_coordinates
from cases import assert_unstructured
from cases import cutout_states
from cases import flat_coordinates
from cases import lookup_of

STRUCTURED_GRID_TYPES = ("reduced_gg", "regular_ll")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_template(provider: MirTemplatesProvider, case: TemplateCase):
    lookup = lookup_of(case)
    state = case.state() if case.state is not None else None
    kwargs = {} if state is None else {"state": state}

    if case.exception is not None:
        with pytest.raises(case.exception, match=case.match):
            provider.template(case.variable, lookup, **kwargs)
        return

    template = provider.template(case.variable, lookup, **kwargs)

    if case.returns_none:
        assert template is None
        return

    assert template is not None, f"No template for {case.id}"

    if case.grid_type is not None:
        assert template.metadata("gridType") == case.grid_type
    if state is not None:
        assert template.metadata("numberOfDataPoints") == len(flat_coordinates(state)[0])
    elif case.npoints is not None:
        assert template.metadata("numberOfDataPoints") == case.npoints
    # The geography is only read when needed: ecCodes cannot build a gridSpec for a Lambert grid
    if case.mars_grid is not None:
        assert template.metadata().geography.mars_grid() == case.mars_grid
    if case.mars_area is not None:
        assert template.metadata().geography.mars_area() == case.mars_area

    if case.header is not None:
        for key, value in case.header.items():
            assert template.metadata(key) == pytest.approx(value, abs=1e-9), key

    if state is not None and case.coordinates:
        assert_coordinates(template, state)
    if state is not None and case.grid_type == "unstructured_grid":
        assert_unstructured(template, state)
    if case.grid_type in STRUCTURED_GRID_TYPES:
        # A structured grid is not tied to a set of points
        assert template.metadata("uuidOfHGrid", default=None) in (None, NULL_UID)


@pytest.mark.parametrize("grid", [None, {"latitudes": [1.0], "longitudes": [1.0]}, 1.0])
def test_grid_size_of_unsupported_grid_raises(grid: Any):
    """``_grid_size`` takes structured grids only (it imports MIR first, so it is not tested without)."""
    with pytest.raises(ValueError, match="Unsupported grid specification"):
        MirTemplatesProvider._grid_size(grid, None)


@pytest.mark.parametrize(
    ("grid", "area", "npoints"),
    [("o96", None, 40320), ("O96", None, 40320), ([1.0, 1.0], None, 181 * 360), ("0.5/0.5", "2/4/2/4", 1)],
)
def test_grid_size(grid: Any, area: Any, npoints: int):
    assert MirTemplatesProvider._grid_size(grid, area) == npoints


@pytest.mark.parametrize("layout", LAYOUTS)
def test_unstructured_templates_differ_by_points(provider: MirTemplatesProvider, layout: str):
    """The full cutout and its global part, with the same lookup, are different unstructured grids."""
    lookup = {"levtype": "sfc", "grid": LAYOUTS[layout].grid}
    states = cutout_states(layout)

    full = provider.template("2t", lookup, state=states["full"])
    glob = provider.template("2t", lookup, state=states["global"])

    assert full.metadata("numberOfDataPoints") != glob.metadata("numberOfDataPoints")
    assert full.metadata("uuidOfHGrid") != glob.metadata("uuidOfHGrid")


def test_layouts_have_different_uids(provider: MirTemplatesProvider):
    uids = {}
    for name, layout in LAYOUTS.items():
        template = provider.template("2t", {"levtype": "sfc", "grid": layout.grid}, state=cutout_states(name)["full"])
        uids[name] = template.metadata("uuidOfHGrid")

    assert len(set(uids.values())) == len(LAYOUTS)
