# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the hindcast date (hdate) logic in MultioOutputPlugin.

When hindcast_reference_date is set, it replaces the date in output metadata,
and the original reference date is preserved as the hdate key.
"""

from datetime import datetime
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
from anemoi.plugins.ecmwf.inference.multio.multio_output import UserDefinedMetadata

CONFIGS_DIR = Path(__file__).parent / "configs"

# ---------------------------------------------------------------------------
# UserDefinedMetadata validation
# ---------------------------------------------------------------------------


class TestUserDefinedMetadataHdate:
    """Tests for hindcast_reference_date on UserDefinedMetadata."""

    BASE_KWARGS = {
        "stream": "oper",
        "type": "fc",
        "class": "od",
        "expver": "1",
    }

    def test_accepts_datetime(self):
        meta = UserDefinedMetadata(**self.BASE_KWARGS, hindcast_reference_date=datetime(2026, 1, 1))
        assert meta.hindcast_reference_date == datetime(2026, 1, 1)

    def test_accepts_none(self):
        meta = UserDefinedMetadata(**self.BASE_KWARGS, hindcast_reference_date=None)
        assert meta.hindcast_reference_date is None

    def test_accepts_yyyymmdd_string(self):
        meta = UserDefinedMetadata(**self.BASE_KWARGS, hindcast_reference_date="20260610")
        assert meta.hindcast_reference_date == datetime(2026, 6, 10)

    def test_accepts_yyyymmdd_int(self):
        meta = UserDefinedMetadata(**self.BASE_KWARGS, hindcast_reference_date=20260610)
        assert meta.hindcast_reference_date == datetime(2026, 6, 10)

    def test_rejects_short_string(self):
        with pytest.raises(ValueError):
            UserDefinedMetadata(**self.BASE_KWARGS, hindcast_reference_date="26")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_multio_server():
    """Mock the MultIO server so write_field calls can be inspected."""
    mocked_server = MagicMock()
    mocked_server.write_field = MagicMock()
    mocked_server.flush = MagicMock()

    with patch("anemoi.plugins.ecmwf.inference.multio.MultioOutputPlugin.open") as mock_open:
        mock_open.side_effect = lambda state: setattr(MultioOutputPlugin, "_server", mocked_server)
        yield mocked_server


def _run_write_step(mock_server, state: State, output_override: dict | None = None) -> list:
    """Create runner, open output, write one step, return write_field call args."""
    overrides = dict(runner="no-model", device="cpu", input="dummy")
    if output_override is not None:
        overrides["output"] = output_override

    config = MockRunConfiguration.load(
        str((CONFIGS_DIR / "multio.yaml").absolute()),
        overrides=overrides,
    )

    runner = create_runner(config)
    output = runner.create_output("data", runner.tensor_handlers["data"].metadata)

    output.open(state)
    output.reference_date = state["date"]
    assert output._server is mock_server

    output.write_step(state)
    return [call.args for call in mock_server.write_field.call_args_list]


def _hdate_output_override(hindcast_reference_date: str) -> dict:
    return {
        "multio": {
            "path": "/dev/null",
            "stream": "eefh",
            "expver": "9999",
            "class": "ai",
            "type": "pf",
            "model": "test",
            "hindcast_reference_date": hindcast_reference_date,
            "number": 0,
            "numberOfForecastsInEnsemble": 51,
        }
    }


# ---------------------------------------------------------------------------
# Integration tests: write_step with hdate
# ---------------------------------------------------------------------------


@fake_checkpoints
@pytest.mark.parametrize(
    "ref_date, hindcast_reference_date, expected_date, expected_hdate",
    [
        (datetime(2025, 6, 10), "20260610", 20260610, 20250610),
        (datetime(2023, 3, 15), "20300315", 20300315, 20230315),
        (datetime(2019, 12, 31), "20251231", 20251231, 20191231),
        (datetime(2024, 2, 28), "20280229", 20280229, 20240228),
        (datetime(2022, 2, 28), "20250228", 20250228, 20220228),
    ],
    ids=["basic", "different-years", "year-end", "leap-target", "non-leap-target"],
)
def test_write_step_sets_date_and_hdate(
    mock_multio_server,
    mock_state,
    ref_date,
    hindcast_reference_date,
    expected_date,
    expected_hdate,
):
    """When hindcast_reference_date is configured, date is replaced and hdate is the original."""
    mock_state["date"] = ref_date
    calls = _run_write_step(mock_multio_server, mock_state, _hdate_output_override(hindcast_reference_date))

    assert len(calls) > 0
    for metadata, field in calls:
        assert metadata["date"] == expected_date
        assert metadata["hdate"] == expected_hdate
        assert isinstance(field, np.ndarray)


@fake_checkpoints
def test_write_step_preserves_time(mock_multio_server, mock_state):
    """The time component from the original reference date is preserved."""
    mock_state["date"] = datetime(2025, 6, 10, 12, 0, 0)
    calls = _run_write_step(mock_multio_server, mock_state, _hdate_output_override("20260610"))

    for metadata, _ in calls:
        assert metadata["date"] == 20260610
        assert metadata["hdate"] == 20250610
        assert metadata["time"] == 120000


@fake_checkpoints
def test_write_step_no_hdate_without_config(mock_multio_server, mock_state):
    """When hindcast_reference_date is not set, hdate is absent from metadata."""
    calls = _run_write_step(mock_multio_server, mock_state)

    for metadata, _ in calls:
        assert "hdate" not in metadata
        assert metadata["date"] == 20200101


@fake_checkpoints
def test_write_step_step_unaffected_by_hdate(mock_multio_server, mock_state):
    """Forecast step in metadata is independent of hdate logic."""
    from datetime import timedelta

    mock_state["date"] = datetime(2025, 6, 10)
    mock_state["step"] = timedelta(hours=24)
    calls = _run_write_step(mock_multio_server, mock_state, _hdate_output_override("20260610"))

    for metadata, _ in calls:
        assert metadata["step"] == 24
        assert metadata["date"] == 20260610
        assert metadata["hdate"] == 20250610
