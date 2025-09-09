# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest
from anemoi.inference.outputs import create_output
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing.mock_checkpoint import MockRunConfiguration

FAKE_METADATA_KEYS = {
    "stream": "test",
    "expver": "1",
    "class": "test",
    "type": "test",
    "model": "test",
}


@pytest.mark.parametrize(
    "plugin, kwargs",
    [
        ("multio.grib", {"path": __file__}),
        ("multio.fdb", {"fdb_config": {}}),
        ("multio.plan", {"plan": None}),
        ("multio", {"fdb_config": {}}),
        ("multio", {"plan": None}),
        ("multio", {"path": __file__}),
    ],
)
@fake_checkpoints
def test_plugin(plugin, kwargs):
    config = MockRunConfiguration.load(
        str((Path(__file__).parent / "configs/simple.yaml").absolute()),
        overrides=dict(output={plugin: {**kwargs, **FAKE_METADATA_KEYS}}),
    )
    runner = create_runner(config)
    output = create_output(runner, config.output)
    assert output is not None


if __name__ == "__main__":
    test_plugin()
