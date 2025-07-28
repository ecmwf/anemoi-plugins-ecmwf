# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import pytest
from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider


class MockTemplateManager:
    pass


VALID_LOOKUPS = [
    {"levtype": "pl"},
    {"levtype": "sfc"},
]


@pytest.fixture
def MirTemplatesProvider_instance():
    """Fixture to create an instance of MirTemplatesProvider."""
    manager = MockTemplateManager()
    return MirTemplatesProvider(manager)


@pytest.mark.parametrize("grid", ["O48", "O96", "N320", "O1280", "1.0/1.0", "0.5/0.5", "0.25/0.25", [0.5, 0.5]])
@pytest.mark.parametrize("lookup", VALID_LOOKUPS)
def test_simple_regrid(MirTemplatesProvider_instance: MirTemplatesProvider, lookup: dict[str, Any], grid: str):
    """Test that the simple regrid template is correctly done."""
    provider = MirTemplatesProvider_instance
    variable = "lsm"

    lookup.update(grid=grid)

    template = provider.template(variable, lookup)

    assert template is not None, f"Template for {variable} with grid {grid} not found"
    expected_grid = grid if "/" not in grid else list(map(float, grid.split("/")))

    assert (
        template.metadata().geography.mars_grid() == expected_grid
    ), f"Expected grid {expected_grid}, got {template.metadata().geography.mars_grid()}"


@pytest.mark.parametrize("grid", ["0.5/0.5", "0.25/0.25", [0.5, 0.5]])
@pytest.mark.parametrize(
    "area_box",
    [
        ("10/-0.5/0.5/11", [10.0, 359.5, 0.5, 11.0]),
        ("2/4/2/4", [2.0, 4.0, 2.0, 4.0]),
        ([2, 4, 2, 4], [2.0, 4.0, 2.0, 4.0]),
    ],
)
@pytest.mark.parametrize("lookup", VALID_LOOKUPS)
def test_regrid_and_area(
    MirTemplatesProvider_instance: MirTemplatesProvider,
    lookup: dict[str, Any],
    grid: str,
    area_box: tuple[str, tuple[int, ...]],
):
    """Test that the regrid and area are correctly applied."""
    provider = MirTemplatesProvider_instance
    variable = "lsm"

    area, box = area_box
    lookup.update(grid=grid, area=area)

    template = provider.template(variable, lookup)

    assert template is not None, f"Template for {variable} with grid {grid} not found"
    expected_grid = grid if "/" not in grid else list(map(float, grid.split("/")))

    assert (
        template.metadata().geography.mars_grid() == expected_grid
    ), f"Expected grid {expected_grid}, got {template.metadata().geography.mars_grid()}"
    assert (
        template.metadata().geography.mars_area() == box
    ), f"Expected box {box}, got {template.metadata().geography.mars_area()}"
