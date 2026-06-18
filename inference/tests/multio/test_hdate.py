# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the hindcast date (hdate) logic in MultioOutputPlugin.

The hdate mechanism swaps the year of the reference date with a
``hindcast_reference_date`` year, writing the original date into the
``hdate`` metadata key.  For example, given reference_date=2025-06-10
and hindcast_reference_date=2026:

    date  -> 20260610   (year replaced)
    hdate -> 20250610   (original date preserved)

Special case: if the reference date is Feb 29, hdate is recorded as
YYYY0228 (clamped to Feb 28) regardless of whether the hindcast year
is a leap year.
"""

from datetime import datetime
from datetime import timedelta
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
from anemoi.plugins.ecmwf.inference.multio.multio_output import _handle_hindcast_date

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

    def test_valid_hindcast_year(self):
        meta = UserDefinedMetadata(**self.BASE_KWARGS, hindcast_reference_date=2026)
        assert meta.hindcast_reference_date == 2026

    def test_hindcast_none_accepted(self):
        meta = UserDefinedMetadata(**self.BASE_KWARGS, hindcast_reference_date=None)
        assert meta.hindcast_reference_date is None

    def test_hindcast_year_too_short(self):
        """A 2-digit year should not be accepted as a valid hindcast year."""
        with pytest.raises(ValueError):
            UserDefinedMetadata(**self.BASE_KWARGS, hindcast_reference_date=26)

    def test_hindcast_year_full_date_rejected_by_length(self):
        """A full YYYYMMDD int (8 digits) should be caught by the 4-digit check."""
        with pytest.raises(ValueError):
            UserDefinedMetadata(**self.BASE_KWARGS, hindcast_reference_date=20260610)


# ---------------------------------------------------------------------------
# _handle_hindcast_date unit tests
# ---------------------------------------------------------------------------


class TestHandleTime:
    """Direct tests for the _handle_hindcast_date function."""

    def test_basic_swap(self):
        """Year is replaced in date, original date preserved as hdate."""
        ref_date, hdate = _handle_hindcast_date(datetime(2025, 6, 10), 2026)
        assert ref_date == datetime(2026, 6, 10)
        assert hdate == 20250610

    def test_preserves_time_component(self):
        ref_date, hdate = _handle_hindcast_date(datetime(2025, 6, 10, 12, 0, 0), 2026)
        assert ref_date == datetime(2026, 6, 10, 12, 0, 0)
        assert hdate == 20250610

    def test_different_year_gap(self):
        ref_date, hdate = _handle_hindcast_date(datetime(2023, 3, 15, 6, 0, 0), 2030)
        assert ref_date == datetime(2030, 3, 15, 6, 0, 0)
        assert hdate == 20230315

    def test_year_end_boundary(self):
        ref_date, hdate = _handle_hindcast_date(datetime(2019, 12, 31, 18, 30, 0), 2025)
        assert ref_date == datetime(2025, 12, 31, 18, 30, 0)
        assert hdate == 20191231

    def test_none_hindcast_returns_original(self):
        """When hindcast_reference_date is None, date is unchanged and hdate is None."""
        original = datetime(2025, 6, 10, 12, 0, 0)
        ref_date, hdate = _handle_hindcast_date(original, None)
        assert ref_date is original
        assert hdate is None

    # --- Feb 29 special case: hdate clamped to 0228 ---

    def test_feb29_to_leap_year_hdate_clamped(self):
        """Feb 29 reference date: hdate recorded as YYYY0228, date swapped normally."""
        ref_date, hdate = _handle_hindcast_date(datetime(2024, 2, 29), 2028)
        assert ref_date == datetime(2028, 2, 29)
        assert hdate == 20280228, f"Expected hdate clamped to 0228, got {hdate}"

    def test_feb29_to_non_leap_year_hdate_clamped(self):
        """Feb 29 reference date with non-leap hindcast year: hdate is YYYY0228.

        The date swap will raise because Feb 29 does not exist in a non-leap year.
        """
        with pytest.raises(ValueError):
            _handle_hindcast_date(datetime(2024, 2, 29), 2025)

    def test_feb28_not_affected_by_clamp(self):
        """Feb 28 should not trigger the Feb 29 special case."""
        ref_date, hdate = _handle_hindcast_date(datetime(2025, 2, 28), 2026)
        assert ref_date == datetime(2026, 2, 28)
        assert hdate == 20250228


# ---------------------------------------------------------------------------
# Helpers and fixtures for integration tests
# ---------------------------------------------------------------------------

STATE_NPOINTS = 50
CONFIGS_DIR = Path(__file__).parent / "configs"


def _make_state(ref_date: datetime, step_hours: int = 6) -> State:
    """Build a mock state with the given reference date."""
    return {
        "latitudes": np.random.uniform(-90, 90, size=STATE_NPOINTS),
        "longitudes": np.random.uniform(-180, 180, size=STATE_NPOINTS),
        "fields": {
            "2t": np.random.uniform(250, 310, size=STATE_NPOINTS),
            "msl": np.random.uniform(500, 1500, size=STATE_NPOINTS),
            "tp": np.random.uniform(0, 10, size=STATE_NPOINTS),
        },
        "date": ref_date,
        "step": timedelta(hours=step_hours),
    }


def _make_hdate_output_override(hindcast_year: int) -> dict:
    """Build an output override dict with hindcast_reference_date set."""
    return {
        "multio": {
            "path": "/dev/null",
            "stream": "eefh",
            "expver": "9999",
            "class": "ai",
            "type": "pf",
            "model": "test",
            "hindcast_reference_date": hindcast_year,
            "number": 0,
            "numberOfForecastsInEnsemble": 51,
        }
    }


def _create_output_and_write(mock_server, state: State, output_override: dict | None = None):
    """Create a runner+output from the test config, open, and write_step.

    Returns the list of (metadata, field) tuples from write_field calls.
    """
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


@pytest.fixture
def mock_multio_server():
    mocked_server = MagicMock()
    mocked_server.write_field = MagicMock()
    mocked_server.flush = MagicMock()

    with patch("anemoi.plugins.ecmwf.inference.multio.MultioOutputPlugin.open") as mock_open:
        mock_open.side_effect = lambda state: setattr(MultioOutputPlugin, "_server", mocked_server)
        yield mocked_server


# ---------------------------------------------------------------------------
# Integration: write_step date/hdate through the real plugin code
# ---------------------------------------------------------------------------


@fake_checkpoints
@pytest.mark.parametrize(
    "ref_date, hindcast_year, expected_date, expected_hdate",
    [
        # Basic: year swapped, original preserved as hdate
        (datetime(2025, 6, 10), 2026, 20260610, 20250610),
        # Different year gap
        (datetime(2023, 3, 15), 2030, 20300315, 20230315),
        # Year-end boundary
        (datetime(2019, 12, 31), 2025, 20251231, 20191231),
        # Leap day into another leap year: hdate clamped to 0228
        (datetime(2024, 2, 29), 2028, 20280229, 20280228),
    ],
    ids=["basic-swap", "different-years", "year-end", "leap-to-leap-clamped"],
)
def test_write_step_hdate_swap(mock_multio_server, ref_date, hindcast_year, expected_date, expected_hdate):
    """write_step should swap the year and preserve the original date as hdate."""
    state = _make_state(ref_date)
    calls = _create_output_and_write(
        mock_multio_server,
        state,
        output_override=_make_hdate_output_override(hindcast_year),
    )

    assert len(calls) > 0, "Expected at least one write_field call"
    for metadata, field in calls:
        assert metadata["date"] == expected_date, f"Expected date {expected_date}, got {metadata['date']}"
        assert metadata["hdate"] == expected_hdate, f"Expected hdate {expected_hdate}, got {metadata['hdate']}"
        assert isinstance(field, np.ndarray)


@fake_checkpoints
def test_write_step_hdate_preserves_time(mock_multio_server):
    """The time component should survive the year swap unchanged."""
    state = _make_state(datetime(2025, 6, 10, 12, 0, 0))
    calls = _create_output_and_write(
        mock_multio_server,
        state,
        output_override=_make_hdate_output_override(2026),
    )

    for metadata, _ in calls:
        assert metadata["date"] == 20260610
        assert metadata["hdate"] == 20250610
        assert metadata["time"] == 120000


@fake_checkpoints
def test_write_step_no_hdate_when_not_configured(mock_multio_server):
    """When hindcast_reference_date is None, hdate should not appear in metadata."""
    state = _make_state(datetime(2025, 6, 10))
    calls = _create_output_and_write(mock_multio_server, state)

    for metadata, _ in calls:
        assert "hdate" not in metadata, f"hdate should not be present when not configured, got {metadata}"
        assert metadata["date"] == 20250610


@fake_checkpoints
def test_write_step_step_field_unaffected_by_hdate(mock_multio_server):
    """The forecast step in metadata should not be affected by the hdate swap."""
    state = _make_state(datetime(2025, 6, 10), step_hours=24)
    calls = _create_output_and_write(
        mock_multio_server,
        state,
        output_override=_make_hdate_output_override(2026),
    )

    for metadata, _ in calls:
        assert metadata["step"] == 24
        assert metadata["date"] == 20260610
        assert metadata["hdate"] == 20250610
