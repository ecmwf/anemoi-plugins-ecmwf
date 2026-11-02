# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid


class MockTemplateManager:
    pass


@pytest.fixture
def provider() -> MirTemplatesProvider:
    return MirTemplatesProvider(MockTemplateManager())


@pytest.fixture
def forward_grids(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record the grid of every MIRRegrid.forward call."""
    forward = MIRRegrid.forward
    grids = []

    def counting_forward(self, fields):
        grids.append(self.grid)
        return forward(self, fields)

    monkeypatch.setattr(MIRRegrid, "forward", counting_forward)
    return grids
