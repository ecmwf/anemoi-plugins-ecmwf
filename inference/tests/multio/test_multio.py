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

import numpy as np
import pytest
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing.mock_checkpoint import MockRunConfiguration
from anemoi.inference.types import State
from anemoi.plugins.ecmwf.inference.multio import MultioOutputPlugin


@pytest.fixture
def mock_multio_server():
    """Fixture that creates and configures a mock MultIO server."""
    mocked_server = MagicMock()
    mocked_write_field = MagicMock()
    mocked_flush = MagicMock()

    mocked_server.attach_mock(mocked_write_field, "write_field")
    mocked_server.attach_mock(mocked_flush, "flush")

    with patch("anemoi.plugins.ecmwf.inference.multio.MultioOutputPlugin.open") as mock_open:
        mock_open.side_effect = lambda state: setattr(MultioOutputPlugin, "_server", mocked_server)
        yield mocked_server


@fake_checkpoints
def test_multio_write_field_called(mock_multio_server, state: State) -> None:
    """Test the write_field method of the MultioOutputPlugin using a fake checkpoint.

    This function creates a MultioOutputPlugin instance, opens it with a mock state,
    and calls the write_field method to ensure that it works as expected.
    """
    # Load configuration
    config = MockRunConfiguration.load(
        str((Path(__file__).parent / "configs/multio.yaml").absolute()),
        overrides=dict(runner="testing", device="cpu", input="dummy"),
    )

    runner = create_runner(config)  # type: ignore
    output = runner.create_output()

    output.open(state)
    output.reference_date = state["date"]
    assert hasattr(output, "_server") and output._server is mock_multio_server

    output.write_step(state)

    # Check that write_field was called with metadata and field data
    mock_multio_server.write_field.assert_called()
    write_field_calls = mock_multio_server.write_field.call_args_list

    # Verify each call has metadata dict and field array
    for call in write_field_calls:
        metadata, field = call.args
        assert isinstance(metadata, dict), "First argument should be metadata dictionary"
        assert isinstance(field, np.ndarray), "Second argument should be numpy array field data"
        assert not np.isnan(field).any(), "Field data contains NaN values"

        # Check that required metadata keys are present
        required_keys = ["param", "levtype", "date", "time", "step", "grid"]
        for key in required_keys:
            assert key in metadata, f"Required metadata key '{key}' missing"

        if metadata["param"] == 228:
            assert "timespan" in metadata, f"Timespan should be set for accumulated fields - {metadata['param']}"
        else:
            assert (
                "timespan" not in metadata
            ), f"Timespan should not be set for non-accumulated fields - {metadata['param']}"

        # Check that missing value is set
        assert "misc-missingValue" in metadata, "Missing value should be set in metadata"

    # Check that flush was called without arguments
    mock_multio_server.flush.assert_called_with()


@fake_checkpoints
def test_multio_workflow_called(mock_multio_server) -> None:
    """Test the inference process using a fake checkpoint.

    This function loads a configuration, creates a runner, and runs the inference
    process to ensure that the system works as expected with the provided configuration.
    """
    # Load configuration
    config = MockRunConfiguration.load(
        str((Path(__file__).parent / "configs/multio.yaml").absolute()),
        overrides=dict(runner="testing", device="cpu", input="dummy"),
    )

    # Create runner and execute
    runner = create_runner(config)  # type: ignore

    # Check calls to the _server property
    assert hasattr(runner.create_output(), "_server")

    runner.execute()

    mock_multio_server.write_field.assert_called()
    mock_multio_server.flush.assert_called()
    mock_multio_server.close_connections.assert_called()


@fake_checkpoints
def test_multio_archiver(mock_multio_server, state) -> None:
    """Test archiver"""

    output_override = {
        "multio": {
            "path": "output.grib",
            "archive_requests": {
                "path": "archive_request.json",
                "extra": {"source": "output.grib"},
                "patch": {"expver": "0001"},
            },
            "stream": "oper",
            "expver": "1",
            "class": "test",
            "type": "test",
            "model": "test",
        }
    }
    # Load configuration
    config = MockRunConfiguration.load(
        str((Path(__file__).parent / "configs/multio.yaml").absolute()),
        overrides=dict(runner="testing", device="cpu", input="dummy", output=output_override),
    )

    # Create runner and execute
    runner = create_runner(config)  # type: ignore
    output = runner.create_output()

    output.open(state)
    output.reference_date = state["date"]
    assert hasattr(output, "_server") and output._server is mock_multio_server

    output.write_step(state)

    assert output._archiver is not None

    archiver = output._archiver
    assert archiver.expect == 3  # Three fields in the state
    request = archiver.request

    assert request["stream"] == ["oper"]
    assert request["expver"] == ["1"]
    assert request["class"] == ["test"]
    assert request["type"] == ["test"]

    assert "param" in request
    assert "levtype" in request
    assert "date" in request
    assert "time" in request
    assert "step" in request
