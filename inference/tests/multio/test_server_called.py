# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing.mock_checkpoint import MockRunConfiguration
from anemoi.plugins.ecmwf.inference.multio import MultioOutputPlugin


@patch("anemoi.plugins.ecmwf.inference.multio.MultioOutputPlugin.open")
@fake_checkpoints
def test_metadata(mock_open_function) -> None:
    """Test the inference process using a fake checkpoint.

    This function loads a configuration, creates a runner, and runs the inference
    process to ensure that the system works as expected with the provided configuration.
    """
    # Mock the _server property
    mocked_server = MagicMock()
    mocked_write_field = MagicMock()
    mocked_flush = MagicMock()

    mocked_server.attach_mock(mocked_write_field, "write_field")
    mocked_server.attach_mock(mocked_flush, "flush")

    mock_open_function.side_effect = lambda state: setattr(MultioOutputPlugin, "_server", mocked_server)

    # Load configuration
    config = MockRunConfiguration.load(
        (Path(__file__).parent / "configs/multio.yaml").absolute(),
        overrides=dict(runner="testing", device="cpu", input="dummy"),
    )

    # Create runner and execute
    runner = create_runner(config)

    # Check calls to the _server property
    assert hasattr(runner.create_output(), "_server")

    runner.execute()

    mocked_write_field.assert_called()
    mocked_flush.assert_called()
