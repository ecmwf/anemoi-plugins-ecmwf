# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from datetime import datetime
from datetime import timedelta

import numpy as np
import pytest
from anemoi.inference.types import State

STATE_NPOINTS = 50


@pytest.fixture
def state() -> State:
    """Fixture to create a mock state for testing."""

    return {
        "latitudes": np.random.uniform(-90, 90, size=STATE_NPOINTS),
        "longitudes": np.random.uniform(-180, 180, size=STATE_NPOINTS),
        "fields": {
            "2t": np.random.uniform(250, 310, size=STATE_NPOINTS),
            "msl": np.random.uniform(500, 1500, size=STATE_NPOINTS),
            "tp": np.random.uniform(0, 10, size=STATE_NPOINTS),
        },
        "date": datetime(2020, 1, 1, 0, 0),
        "step": timedelta(hours=6),
    }
