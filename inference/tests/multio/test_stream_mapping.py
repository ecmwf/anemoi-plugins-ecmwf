# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the stream mapping logic in MultioOutputPlugin.

When the user-defined stream is ``ORIGIN``, the stream written for each field is
taken from the variable's GRIB keys (default ``oper``). It is then remapped to
the ensemble stream if ``number`` is set, and to the hindcast stream if
``hindcast_reference_date`` is set.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing.mock_checkpoint import MockRunConfiguration
from anemoi.inference.types import State

from anemoi.plugins.ecmwf.inference.multio import MultioOutputPlugin
from anemoi.plugins.ecmwf.inference.multio.multio_output import UserDefinedMetadata

CONFIGS_DIR = Path(__file__).parent / "configs"

BASE_USER_METADATA = {
    "type": "fc",
    "class": "ai",
    "expver": "9999",
    "model": "test",
}
ENSEMBLE_METADATA = {"number": 3, "numberOfForecastsInEnsemble": 51}
HINDCAST_METADATA = {"hindcast_reference_date": "20260610"}

# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def _plugin_with_user_metadata(**user_metadata: Any) -> MultioOutputPlugin:
    """Create a bare plugin carrying only the user defined metadata."""
    plugin = object.__new__(MultioOutputPlugin)
    plugin._user_defined_metadata = UserDefinedMetadata(**{"stream": "ORIGIN", **BASE_USER_METADATA, **user_metadata})
    return plugin


@pytest.mark.parametrize(
    "extra, expected_ensemble, expected_hindcast",
    [
        ({}, False, False),
        (ENSEMBLE_METADATA, True, False),
        (HINDCAST_METADATA, False, True),
        ({**ENSEMBLE_METADATA, **HINDCAST_METADATA}, True, True),
        ({"number": 0, "numberOfForecastsInEnsemble": 51}, True, False),  # number 0 is still an ensemble member
    ],
    ids=["deterministic", "ensemble", "hindcast", "ensemble-hindcast", "ensemble-number-zero"],
)
def test_is_ensemble_and_is_hindcast(extra, expected_ensemble, expected_hindcast):
    plugin = _plugin_with_user_metadata(**extra)
    assert plugin.is_ensemble is expected_ensemble
    assert plugin.is_hindcast is expected_hindcast


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_multio_server():
    """Mock the MultIO server so write_field calls can be inspected."""
    mocked_server = MagicMock()

    with patch("anemoi.plugins.ecmwf.inference.multio.MultioOutputPlugin.open") as mock_open:
        mock_open.side_effect = lambda state: setattr(MultioOutputPlugin, "_server", mocked_server)
        yield mocked_server


class _VariableWithStream:
    """Proxy around a typed variable whose GRIB keys carry a given stream."""

    def __init__(self, variable: Any, stream: str | None) -> None:
        self._variable = variable
        self._stream = stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self._variable, name)

    @property
    def grib_keys(self) -> dict[str, Any]:
        keys = dict(self._variable.grib_keys)
        if self._stream is not None:
            keys["stream"] = self._stream
        return keys


def _written_metadata(
    mock_server: MagicMock,
    state: State,
    user_metadata: dict[str, Any],
    variable_stream: str | None,
) -> list[dict[str, Any]]:
    """Write one step and return the metadata dicts passed to ``write_field``."""
    output_override = {"multio": {"path": "/dev/null", **BASE_USER_METADATA, **user_metadata}}
    config = MockRunConfiguration.load(
        str((CONFIGS_DIR / "multio.yaml").absolute()),
        overrides=dict(runner="no-model", device="cpu", input="dummy", output=output_override),
    )

    runner = create_runner(config)
    output = runner.create_output("data", runner.tensor_handlers["data"].metadata)

    output.typed_variables = {
        name: _VariableWithStream(variable, variable_stream) for name, variable in output.typed_variables.items()
    }

    output.open(state)
    output.reference_date = state["date"]
    assert output._server is mock_server

    output.write_step(state)

    metadatas = [call.args[0] for call in mock_server.write_field.call_args_list]
    assert len(metadatas) > 0
    return metadatas


# ---------------------------------------------------------------------------
# write_step: stream ends up in the metadata sent to multio
# ---------------------------------------------------------------------------


@fake_checkpoints
@pytest.mark.parametrize(
    "extra, variable_stream, expected",
    [
        # Deterministic: unchanged
        ({}, None, "oper"),
        ({}, "oper", "oper"),
        ({}, "wave", "wave"),
        # Ensemble
        (ENSEMBLE_METADATA, None, "enfo"),
        (ENSEMBLE_METADATA, "oper", "enfo"),
        (ENSEMBLE_METADATA, "wave", "waef"),
        # Ensemble + hindcast
        ({**ENSEMBLE_METADATA, **HINDCAST_METADATA}, None, "enfh"),
        ({**ENSEMBLE_METADATA, **HINDCAST_METADATA}, "oper", "enfh"),
        ({**ENSEMBLE_METADATA, **HINDCAST_METADATA}, "wave", "enwh"),
        # Hindcast without ensemble: oper is not remapped
        (HINDCAST_METADATA, None, "oper"),
        (HINDCAST_METADATA, "oper", "oper"),
        (HINDCAST_METADATA, "wave", "wave"),
        # Unknown streams pass through unchanged
        ({}, "scda", "scda"),
        (ENSEMBLE_METADATA, "scda", "scda"),
        ({**ENSEMBLE_METADATA, **HINDCAST_METADATA}, "scda", "scda"),
        # Streams that are already ensemble streams are not remapped by the ensemble table
        (ENSEMBLE_METADATA, "enfo", "enfo"),
        (ENSEMBLE_METADATA, "waef", "waef"),
    ],
    ids=[
        "deterministic-default",
        "deterministic-oper",
        "deterministic-wave",
        "ensemble-default",
        "ensemble-oper",
        "ensemble-wave",
        "ensemble-hindcast-default",
        "ensemble-hindcast-oper",
        "ensemble-hindcast-wave",
        "hindcast-only-default",
        "hindcast-only-oper",
        "hindcast-only-wave",
        "deterministic-unknown",
        "ensemble-unknown",
        "ensemble-hindcast-unknown",
        "ensemble-already-enfo",
        "ensemble-already-waef",
    ],
)
def test_origin_stream_mapping(mock_multio_server, mock_state, extra, variable_stream, expected):
    """With ``stream: ORIGIN`` the stream comes from the variable, remapped for ensemble/hindcast."""
    metadatas = _written_metadata(
        mock_multio_server, mock_state, {"stream": "ORIGIN", **extra}, variable_stream=variable_stream
    )

    for metadata in metadatas:
        assert metadata["stream"] == expected


@fake_checkpoints
@pytest.mark.parametrize(
    "extra",
    [{}, ENSEMBLE_METADATA, {**ENSEMBLE_METADATA, **HINDCAST_METADATA}],
    ids=["deterministic", "ensemble", "ensemble-hindcast"],
)
@pytest.mark.parametrize("variable_stream", [None, "oper", "wave"])
@pytest.mark.parametrize("user_stream", ["oper", "enfo", "eefh"])
def test_non_origin_stream_not_overridden(mock_multio_server, mock_state, extra, variable_stream, user_stream):
    """A user-defined stream other than ORIGIN is never written per field, nor remapped."""
    metadatas = _written_metadata(
        mock_multio_server, mock_state, {"stream": user_stream, **extra}, variable_stream=variable_stream
    )

    for metadata in metadatas:
        assert "stream" not in metadata
