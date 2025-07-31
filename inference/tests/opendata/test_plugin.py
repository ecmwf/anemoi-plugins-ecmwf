# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

from anemoi.inference.inputs import create_input
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing.mock_checkpoint import MockRunConfiguration


@fake_checkpoints
def test_plugin():
    config = MockRunConfiguration.load(
        (Path(__file__).parent / "configs/simple.yaml").absolute(),
        overrides=dict(input="opendata"),
    )
    runner = create_runner(config)
    input = create_input(runner, config.input)
    assert input is not None


if __name__ == "__main__":
    test_plugin()
