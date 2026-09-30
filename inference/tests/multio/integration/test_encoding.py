# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Integration tests: real multio encoding -> GRIB -> earthkit header check.

Unlike ``tests/multio/test_multio.py`` (which mocks the multio server), this
suite drives the **real** multio output plugin with the mtg2 encoder
(``encode-mtg2``), writes actual GRIB messages to disk, reads them back with
earthkit-data / eccodes, and asserts the encoded GRIB header keys.

The sweep dimensions (param, levtype, levelist, step, accumulation window,
reference time, reference date, stream oper/enfo, ensemble number) are all
declared in the YAML files under ``cases/``. To add an edge case, add an entry
there -- no Python changes required.

Each case runs in its own subprocess (see ``_worker.py``) so multio's internal
encoder state cannot bleed between cases.

Skips gracefully if pymultio / earthkit-data are not installed; run with
``pytest -m integration`` (and ``-p no:cacheprovider`` if desired).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from .conftest import load_cases
from .schema import Case

# ---------------------------------------------------------------------------
# Dependency gate -- these are heavy, compiled, optional dependencies.
# ---------------------------------------------------------------------------
multio = pytest.importorskip("multio", reason="pymultio not installed")
ekd = pytest.importorskip("earthkit.data", reason="earthkit-data not installed")
eccodes = pytest.importorskip("eccodes", reason="eccodes not installed")

pytestmark = pytest.mark.integration

WORKER = "tests.multio.integration._worker"

ALL_CASES = load_cases()


def _multiolib_version() -> str:
    """Version of the compiled ``multiolib`` (the encoder), not the bindings.

    ``pymultio`` (the Python bindings) and ``multiolib`` (the compiled library
    that actually does the encoding) are versioned independently and can diverge
    on the package index -- e.g. on the macOS CI runners we see
    ``pymultio==2.11.x`` alongside ``multiolib==2.10.x``. Since the encoding
    behaviour we assert on lives in ``multiolib``, gate cases on *its* version.
    """
    import importlib.metadata as md

    try:
        return md.version("multiolib")
    except md.PackageNotFoundError:  # pragma: no cover - importorskip handles this
        return "0"


MULTIOLIB_VERSION = _multiolib_version()


def _skip_if_multio_too_old(case: Case) -> None:
    """Skip a case whose expectations require a newer multiolib than installed.

    Encoder behaviour is version-sensitive: e.g. multiolib 2.10 encodes the
    ``time`` metadata as hours (1200 -> 12) and rejects the ``timespan: fs``
    accumulation path, both of which changed in 2.11. Rather than assert
    version-specific values, such cases declare ``min_multio`` and are skipped
    on older builds (as seen on the macOS CI runners that lag the linux index).
    """
    if case.min_multio is None:
        return
    from packaging.version import Version

    if Version(MULTIOLIB_VERSION) < Version(case.min_multio):
        pytest.skip(f"requires multiolib >= {case.min_multio} (installed {MULTIOLIB_VERSION})")


def _run_worker(case: Case, tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """Run one case in an isolated subprocess; return (worker result, grib path).

    Output is deliberately capped when surfaced in assertions: a multio encode
    failure emits a very large (recursively logged) C++ backtrace, so we only
    ever show a bounded tail of stderr.
    """
    case_json = tmp_path / "case.json"
    out_grib = tmp_path / "out.grib"
    stdout_path = tmp_path / "worker.out"
    stderr_path = tmp_path / "worker.err"
    case_json.write_text(json.dumps(case.to_worker_dict()))

    # Redirect worker stdout/stderr to files rather than in-memory pipes. A
    # failed mtg2 encode emits an unbounded, recursively-logged C++ backtrace on
    # stderr; capturing that into a pipe would exhaust memory. Writing to a file
    # (and only reading a bounded tail) keeps the parent safe.
    with open(stdout_path, "w") as out_fh, open(stderr_path, "w") as err_fh:
        proc = subprocess.run(
            [sys.executable, "-m", WORKER, str(case_json), str(out_grib)],
            stdout=out_fh,
            stderr=err_fh,
            text=True,
            cwd=str(Path(__file__).parents[3]),  # repo root so ``tests`` is importable
        )

    result: dict[str, Any] = {"ok": False, "error": "worker produced no RESULT line"}
    for line in stdout_path.read_text().splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT ") :])
            break

    if proc.returncode != 0 and result.get("ok"):
        tail = "\n".join(_tail(stderr_path, 20))
        result = {"ok": False, "error": f"worker exited {proc.returncode}\n{tail}"}

    return result, out_grib


def _tail(path: Path, n: int) -> list[str]:
    """Read at most the last ``n`` lines of a (possibly huge) file, bounded."""
    lines: list[str] = []
    with open(path) as handle:
        for line in handle:
            lines.append(line.rstrip("\n"))
            if len(lines) > n:
                lines.pop(0)
    return lines


def _read_headers(grib_path: Path, keys: list[str]) -> list[dict[str, Any]]:
    """Read requested GRIB header keys from every message using eccodes."""
    messages: list[dict[str, Any]] = []
    with open(grib_path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                record: dict[str, Any] = {}
                for key in keys:
                    try:
                        record[key] = eccodes.codes_get(gid, key)
                    except Exception:  # noqa: BLE001 - missing key surfaces as assertion
                        record[key] = None
                messages.append(record)
            finally:
                eccodes.codes_release(gid)
    return messages


def _check_message(header: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    """Return a list of human-readable mismatches (empty == all good)."""
    mismatches = []
    for key, want in expect.items():
        got = header.get(key)
        # eccodes returns numbers as int/float and strings as str; normalise the
        # comparison so YAML ints match eccodes ints and YAML strings match keys
        # like stepRange ("0-6").
        if isinstance(want, str) and not isinstance(got, str):
            got_cmp: Any = str(got)
        else:
            got_cmp = got
        if got_cmp != want:
            mismatches.append(f"  {key}: expected {want!r}, got {got!r}")
    return mismatches


@pytest.mark.parametrize("case", ALL_CASES, ids=[c.id for c in ALL_CASES])
def test_multio_grib_encoding(case: Case, tmp_path: Path) -> None:
    """Encode with real multio and verify the GRIB header round-trip.

    Two shapes of case are supported:

    - single message: ``expect`` maps GRIB keys -> expected values, exactly one
      message must be written;
    - sequence: ``expect_sequence`` is a list of such maps, one per forecast step
      in ``steps`` (used to catch the ``cached: true`` stepRange-collapse bug).

    A case may set ``xfail: "<reason>"`` to mark a known limitation: the test
    then xfails on encode failure or header mismatch, and xpasses (loudly) once
    the behaviour is fixed so the marker can be removed.
    """
    _skip_if_multio_too_old(case)

    xfail_reason = case.xfail
    expect_sequence = case.expect_sequence
    expect = case.expect

    result, out_grib = _run_worker(case, tmp_path)

    def _finish(all_mismatches: list[str]) -> None:
        if xfail_reason:
            if all_mismatches or not result["ok"]:
                reason = result["error"] if not result["ok"] else "; ".join(all_mismatches)
                pytest.xfail(f"{xfail_reason} :: {reason[:300]}")
            # No mismatch and encode ok -> the bug is fixed; let it xpass-fail
            # (strict) so the marker gets removed.
            return
        assert result["ok"], f"multio encode failed for {case.id}:\n{result['error'][:1000]}"
        assert not all_mismatches, "GRIB header mismatch for {}:\n{}".format(case.id, "\n".join(all_mismatches))

    # If the encode failed and this is a known xfail, short-circuit before trying
    # to read a (possibly absent) GRIB file.
    if not result["ok"] and xfail_reason:
        _finish([])
        return

    assert result["ok"], f"multio encode failed for {case.id}:\n{result['error'][:1000]}"
    assert out_grib.exists() and out_grib.stat().st_size > 0, "no GRIB written"

    if expect_sequence is not None:
        keys = sorted({k for step_expect in expect_sequence for k in step_expect})
        messages = _read_headers(out_grib, keys)
        mismatches: list[str] = []
        if len(messages) != len(expect_sequence):
            mismatches.append(f"  expected {len(expect_sequence)} messages, got {len(messages)}")
        for idx, (header, step_expect) in enumerate(zip(messages, expect_sequence)):
            for m in _check_message(header, step_expect):
                mismatches.append(f"  [step #{idx}]{m}")
        _finish(mismatches)
        return

    messages = _read_headers(out_grib, list(expect.keys()))
    mismatches = []
    if len(messages) != 1:
        mismatches.append(f"  expected exactly one message, got {len(messages)}")
    else:
        mismatches = _check_message(messages[0], expect)
    _finish(mismatches)
