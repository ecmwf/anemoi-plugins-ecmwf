# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Subprocess worker that drives the *real* multio output plugin for one case.

Why a subprocess?
-----------------
The multio/mtg2 encoder keeps internal (statistics / product-definition) state
across ``multio.Multio()`` instances *inside the same process*. Running several
cases in one interpreter causes that state to bleed between them: e.g. the
forecast ``step`` / ``stepRange`` of the first accumulation field gets "stuck"
and is reused for later fields (observed as every ``tp`` collapsing to the same
``stepRange``). To guarantee each case is encoded from a clean slate we run one
case per subprocess.

Protocol
--------
``python -m tests.multio.integration._worker <case.json> <out.grib>``

- reads the resolved case description (JSON) from the first argument;
- writes exactly one GRIB file at the second argument via a real multio plan
  ``[EncodeMTG (mtg2) -> File sink]``;
- prints a single JSON line to stdout: ``{"ok": bool, "error": str|None}``.

The parent (pytest) then reads the GRIB back with earthkit/eccodes and asserts
on the encoded header keys. Keeping *verification* in the parent means the
worker only needs multio + numpy, and all assertion logic lives in the test.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np


def _parse_step(value: int | str) -> timedelta:
    """Parse a case ``step`` into a timedelta.

    Accepts whole hours as an int/int-like, or a metkit-style duration string
    with an explicit unit suffix: ``s`` (seconds), ``m`` (minutes),
    ``h`` (hours), ``d`` (days). This mirrors what an anemoi runner would hand
    to the output as ``state["step"]``.
    """
    if isinstance(value, (int, float)):
        return timedelta(hours=float(value))
    text = str(value).strip()
    if text.isdigit():
        return timedelta(hours=int(text))
    unit = text[-1]
    amount = float(text[:-1])
    factor = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return timedelta(seconds=amount * factor)


def _npoints(grid: str) -> int:
    """Number of grid points for the small gaussian grids used in tests.

    multio (via atlas) derives the geometry from the ``grid`` key, so the
    payload length must match exactly. Only the handful of grids used by the
    test configs need to be listed here; add new ones as cases require.
    """
    counts = {
        "O32": 5248,
        "O48": 6480,
        "N32": 6114,
    }
    key = grid.upper()
    if key not in counts:
        raise KeyError(f"Unknown grid {grid!r}; add its point count to tests/multio/integration/_worker.py:_npoints")
    return counts[key]


def _build_plugin(case: dict[str, Any], out_grib: str):
    """Construct a real MultioOutputGribPlugin without the heavy Output.__init__.

    We inject exactly the attributes ``write_step`` touches (metadata, context,
    typed_variables, user metadata, plan) so the test drives the genuine encode
    path with a real multio server rather than a mock.
    """
    import multio
    from anemoi.inference.post_processors.accumulate import Accumulate
    from anemoi.plugins.ecmwf.inference.multio.multio_output import MultioOutputGribPlugin
    from anemoi.plugins.ecmwf.inference.multio.multio_output import UserDefinedMetadata
    from anemoi.transform.variables import Variable

    plugin = MultioOutputGribPlugin.__new__(MultioOutputGribPlugin)

    # ``cached`` mirrors the operational plan's ``encode-mtg2: {cached: true}``.
    # With caching on, mtg2 reuses a GRIB template across messages, which freezes
    # the first accumulation field's time-range for all later ones -> the
    # observed collapse of every ``tp`` to ``stepRange 0-1``. Default False.
    cached = bool(case.get("cached", False))

    plan = multio.plans.Client(
        plans=[
            multio.plans.Plan(
                name="integration-test",
                actions=[
                    multio.plans.EncodeMTG(cached=cached),  # plan action: encode-mtg2
                    multio.plans.Sink(
                        sinks=[
                            multio.plans.sinks.File(
                                append=True,
                                per_server=False,
                                path=out_grib,
                            )
                        ]
                    ),
                ],
            )
        ]
    )
    plugin._plan = plan
    plugin._archiver = None
    plugin._initial_state_diagnostics_grib = None
    plugin._server = None
    plugin._user_defined_metadata = UserDefinedMetadata(**case["user"])

    grid = case["grid"]
    plugin.metadata = SimpleNamespace(grid=grid, timestep=_parse_step(case.get("timestep", 6)))

    post_processors: dict[str, list[Any]] = {}
    if case.get("accumulate_from_start"):
        post_processors = {"default": [Accumulate.__new__(Accumulate)]}

    ref = case["reference"]
    ref_dt = datetime.strptime(f"{ref['date']:08d}{int(ref['time']):04d}", "%Y%m%d%H%M")
    plugin.context = SimpleNamespace(post_processors=post_processors, reference_date=ref_dt)
    plugin.reference_date = ref_dt

    vspec = dict(case["variable"])
    vname = vspec.pop("name")
    variable = Variable.from_dict(vname, vspec)
    plugin.typed_variables = {vname: variable}

    return plugin, vname, grid


def _make_field(case: dict[str, Any], npoints: int) -> np.ndarray:
    """Generate the field payload, optionally injecting NaNs for bitmap tests."""
    field_cfg = case.get("field") or {}
    low, high = field_cfg.get("fill", [250.0, 310.0])
    rng = np.random.default_rng(0)
    field = rng.uniform(low, high, size=npoints).astype("float32")

    nan_fraction = float(field_cfg.get("nan_fraction", 0.0))
    if nan_fraction > 0.0:
        n_nan = int(npoints * nan_fraction)
        field[rng.choice(npoints, size=n_nan, replace=False)] = np.nan
    return field


def run_case(case: dict[str, Any], out_grib: str) -> None:
    """Write one or more forecast steps through a real multio server.

    A case may specify either a single ``step`` (one message) or a list of
    ``steps`` (a sequence written through the *same* multio session). The
    multi-step form is what exposes the operational ``cached: true`` regression:
    successive accumulation fields must encode distinct ``stepRange`` values and
    not collapse to the first one.

    IMPORTANT: if a write raises we must NOT call ``plugin.close()``. On a failed
    message multio's ``FailureAware`` handler re-runs during
    ``flush``/``close_connections`` and dumps a large, recursively-logged C++
    backtrace to stderr without bound -- captured by the parent this exhausts
    memory. So on failure we re-raise immediately; ``main`` reports the error
    then hard-exits before any multio teardown.
    """
    plugin, vname, grid = _build_plugin(case, out_grib)
    field = _make_field(case, _npoints(grid))
    npoints = field.size

    steps = case["steps"] if case.get("steps") is not None else [case["step"]]
    steps_td = [_parse_step(s) for s in steps]

    first_state = {
        "date": plugin.reference_date,
        "step": steps_td[0],
        "fields": {vname: field},
    }
    plugin.open(first_state)

    for step_td in steps_td:
        state = {
            "date": plugin.reference_date,
            "step": step_td,
            # fresh values per step; new field each time to mimic a real forecast
            "fields": {vname: _make_field(case, npoints)},
        }
        plugin.write_step(state)  # may raise -- deliberately no close() on failure

    plugin.close()


def main(argv: list[str]) -> int:
    import os

    case_path, out_grib = argv[1], argv[2]
    with open(case_path) as handle:
        case = json.load(handle)

    result: dict[str, Any] = {"ok": True, "error": None}
    try:
        run_case(case, out_grib)
    except BaseException as exc:  # noqa: BLE001 - relayed to parent as data
        # Truncate: a multio/mtg2 failure message can be enormous (nested
        # FailureAware contexts). Keep only the first, most useful line(s).
        message = str(exc).strip().splitlines()
        head = " | ".join(message[:3])[:500]
        result = {"ok": False, "error": f"{type(exc).__name__}: {head}"}

    # Single machine-readable line for the parent to parse.
    sys.stdout.write("RESULT " + json.dumps(result) + "\n")
    sys.stdout.flush()

    # Hard exit: skip interpreter shutdown / multio atexit teardown, which on a
    # failed encode would re-invoke the FailureAware handler and flood stderr
    # with an unbounded, recursive C++ backtrace (exhausting the parent's memory
    # when stderr is captured). ``os._exit`` bypasses all of that.
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv)
