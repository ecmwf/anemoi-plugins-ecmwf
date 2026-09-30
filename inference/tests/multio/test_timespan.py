# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``timespan`` metadata logic in MultioOutputPlugin.

The time span describes the period a field represents and depends on its time
processing (see the ECMWF parameter and GRIB2 statistical process code tables):

- instantaneous fields (e.g. ``2t`` paramId 167, ``msl`` paramId 151) have no
  time span;
- accumulations (e.g. ``tp`` paramId 228) span their accumulation period, or
  are flagged ``"fs"`` (from start) when accumulated from the forecast start;
- non-instantaneous statistical fields (e.g. the 10 metre wind gust ``10fg``
  paramId 49, a maximum over the preceding period) span their processing
  period.
"""

from datetime import timedelta
from types import SimpleNamespace

import pytest
from anemoi.plugins.ecmwf.inference.multio.multio_output import MultioOutputPlugin
from anemoi.plugins.ecmwf.inference.multio.multio_output import _format_timespan
from anemoi.transform.variables import Variable

# ---------------------------------------------------------------------------
# Variable fixtures mirroring real ECMWF fields
# ---------------------------------------------------------------------------


def _variable(name: str, data: dict) -> Variable:
    return Variable.from_dict(name, data)


# Instantaneous fields -> no time span
T2M = _variable("2t", {"mars": {"param": "2t", "levtype": "sfc"}})
MSL = _variable("msl", {"mars": {"param": "msl", "levtype": "sfc"}})

# Accumulation with an explicit period (e.g. 6-hourly total precipitation)
TP_PERIOD = _variable(
    "tp",
    {
        "process": "accumulation",
        "period": [0, 6],
        "mars": {"param": "tp", "levtype": "sfc"},
    },
)

# Accumulation without a period recorded in the checkpoint metadata
TP_NO_PERIOD = _variable(
    "tp",
    {"process": "accumulation", "mars": {"param": "tp", "levtype": "sfc"}},
)

# Non-instantaneous statistical field: 10 metre wind gust (maximum over period)
WIND_GUST_6H = _variable(
    "10fg",
    {
        "process": "maximum",
        "period": [0, 6],
        "mars": {"param": "10fg", "levtype": "sfc"},
    },
)

# Non-instantaneous statistical field with sub-hourly (30 min) period
WIND_GUST_30M = _variable(
    "10fg",
    {
        "process": "maximum",
        "period": ["0h", "30m"],
        "mars": {"param": "10fg", "levtype": "sfc"},
    },
)


def _make_plugin(*, accumulated_from_start: bool, timestep: timedelta) -> MultioOutputPlugin:
    """Build a bare plugin instance suitable for calling ``_timespan_for``.

    ``_timespan_for`` only touches ``self.context.post_processors`` (via
    ``_is_accumulated_from_start``) and ``self.metadata.timestep``, so we avoid
    the heavy ``__init__`` and inject just those attributes.
    """
    plugin = MultioOutputPlugin.__new__(MultioOutputPlugin)
    post_processors = {"default": [object()]}
    if accumulated_from_start:
        from anemoi.inference.post_processors.accumulate import Accumulate

        post_processors = {"default": [Accumulate.__new__(Accumulate)]}
    plugin.context = SimpleNamespace(post_processors=post_processors)
    plugin.metadata = SimpleNamespace(timestep=timestep)
    return plugin


# ---------------------------------------------------------------------------
# _format_timespan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "period, expected",
    [
        (timedelta(hours=6), 6),
        (timedelta(hours=1), 1),
        (timedelta(hours=24), 24),
    ],
)
def test_format_timespan_whole_hours(period, expected):
    assert _format_timespan(period) == expected


@pytest.mark.parametrize(
    "period",
    [
        timedelta(minutes=30),
        timedelta(minutes=90),
        timedelta(seconds=45),
    ],
)
def test_format_timespan_rejects_sub_hourly(period):
    # Sub-hourly spans are unsupported by the mtg2 encoder (metkit). The plugin
    # must refuse them with a clear Python error rather than emitting an
    # unencodable "<seconds>s" duration string that floods multio's failure log.
    with pytest.raises(ValueError, match="[Ss]ub-hourly"):
        _format_timespan(period)


# ---------------------------------------------------------------------------
# _timespan_for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variable", [T2M, MSL], ids=["2t", "msl"])
def test_instantaneous_fields_have_no_timespan(variable):
    plugin = _make_plugin(accumulated_from_start=False, timestep=timedelta(hours=6))
    assert plugin._timespan_for(variable) is None


def test_accumulation_uses_its_period():
    plugin = _make_plugin(accumulated_from_start=False, timestep=timedelta(hours=1))
    assert plugin._timespan_for(TP_PERIOD) == 6


def test_accumulation_without_period_falls_back_to_timestep():
    plugin = _make_plugin(accumulated_from_start=False, timestep=timedelta(hours=6))
    # No period recorded -> must not crash, falls back to the model timestep.
    assert plugin._timespan_for(TP_NO_PERIOD) == 6


def test_accumulation_from_start_is_flagged_fs():
    plugin = _make_plugin(accumulated_from_start=True, timestep=timedelta(hours=6))
    assert plugin._timespan_for(TP_PERIOD) == "fs"
    assert plugin._timespan_for(TP_NO_PERIOD) == "fs"


def test_non_instantaneous_statistical_field_uses_period():
    plugin = _make_plugin(accumulated_from_start=False, timestep=timedelta(hours=1))
    # 10fg is a maximum over the preceding period, not an accumulation.
    assert plugin._timespan_for(WIND_GUST_6H) == 6


def test_non_instantaneous_subhourly_period_rejected():
    plugin = _make_plugin(accumulated_from_start=False, timestep=timedelta(hours=1))
    # Sub-hourly statistical windows are unsupported by mtg2 and must raise
    # rather than produce an unencodable duration string.
    with pytest.raises(ValueError, match="[Ss]ub-hourly"):
        plugin._timespan_for(WIND_GUST_30M)


def test_non_instantaneous_not_affected_by_accumulate_from_start():
    # The "fs" (from start) flag only applies to accumulations, never to
    # statistical fields such as 10fg.
    plugin = _make_plugin(accumulated_from_start=True, timestep=timedelta(hours=1))
    assert plugin._timespan_for(WIND_GUST_6H) == 6
