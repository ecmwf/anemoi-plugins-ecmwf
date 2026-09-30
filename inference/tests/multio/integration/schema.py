# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pydantic schema for the multio -> GRIB integration test cases.

Each YAML file under ``cases/`` is a :class:`CaseFile`: a ``defaults`` block and
a list of ``cases``. Defaults are deep-merged into every case, then each merged
case is validated as a :class:`Case`. Using a schema (rather than loose dicts)
means a malformed or mistyped case fails fast at collection time with a clear
error, and the accepted knobs are documented in one place.
"""

from __future__ import annotations

import copy
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


class UserMeta(BaseModel):
    """User-defined metadata passed straight to ``UserDefinedMetadata``.

    Mirrors the plugin's own model (class/type/stream/expver/model/number/...).
    ``extra="allow"`` so new plugin keys can be exercised without editing this
    schema, while the common ones are typed for early validation.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    klass: str = Field(alias="class")
    type: str
    stream: str
    expver: str | int
    model: str | None = None
    number: int | None = None
    numberOfForecastsInEnsemble: int | None = None
    generatingProcessIdentifier: int | None = None

    def to_plugin_kwargs(self) -> dict[str, Any]:
        """Serialise back to the ``class``-aliased dict the plugin expects."""
        return self.model_dump(by_alias=True, exclude_none=True)


class Reference(BaseModel):
    """Forecast reference (analysis) date/time."""

    model_config = ConfigDict(extra="forbid")

    date: int
    """Reference date as YYYYMMDD, e.g. 20260923."""
    time: int = 0
    """Reference time as HHMM, e.g. 0 or 1200."""


class Variable(BaseModel):
    """anemoi-transform ``Variable`` spec (mars keys + optional processing)."""

    model_config = ConfigDict(extra="allow")

    name: str
    """Field name used as the state key, e.g. ``2t`` or ``t_500``."""
    mars: dict[str, Any]
    """Mars keys, e.g. ``{param: t, levtype: pl, levelist: 500}``."""
    process: str | None = None
    """Time processing, e.g. ``accumulation`` or ``maximum``. Optional."""
    period: list[Any] | None = None
    """Processing window as ``[start, end]`` (hours or duration strings)."""

    def to_transform_dict(self) -> tuple[str, dict[str, Any]]:
        """Return (name, spec) suitable for ``Variable.from_dict``."""
        spec = self.model_dump(exclude_none=True)
        name = spec.pop("name")
        return name, spec


class FieldGen(BaseModel):
    """Controls for generating the payload values of a case."""

    model_config = ConfigDict(extra="forbid")

    fill: tuple[float, float] = (250.0, 310.0)
    """Uniform ``[low, high]`` range for the generated values."""
    nan_fraction: float = 0.0
    """Fraction of points set to NaN (to exercise bitmap / missing values)."""


class Case(BaseModel):
    """A fully-resolved integration case (defaults already merged in)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    """Unique, human-readable id within its file."""
    grid: str
    """Gaussian grid name, e.g. ``O48`` (drives the payload length)."""
    user: UserMeta
    reference: Reference
    variable: Variable

    # step XOR steps -----------------------------------------------------
    step: int | str | None = None
    """Single forecast step: whole hours (int) or duration string (e.g. ``6``)."""
    steps: list[int | str] | None = None
    """A sequence of steps written through one session (multi-message case)."""

    timestep: int | str = 6
    """Model timestep, used as a fallback accumulation period."""
    accumulate_from_start: bool = False
    """Emulate an ``Accumulate`` post-processor (timespan ``fs``)."""
    cached: bool = False
    """Encode with ``encode-mtg2: {cached: true}`` (operational plan parity)."""

    field: FieldGen = Field(default_factory=FieldGen)

    # expectations: expect XOR expect_sequence ---------------------------
    expect: dict[str, Any] = Field(default_factory=dict)
    """GRIB header keys -> expected values for a single-message case."""
    expect_sequence: list[dict[str, Any]] | None = None
    """One expectation map per step for a multi-message (sequence) case."""

    xfail: str | None = None
    """If set, the case is a known limitation and is expected to fail."""

    min_multio: str | None = None
    """Minimum pymultio version required, e.g. ``"2.11"``.

    Encoder behaviour is version-sensitive (e.g. ``time`` encoding and the
    ``timespan: fs`` accumulation path changed between 2.10 and 2.11), so cases
    whose expectations assume newer behaviour declare the floor and are skipped
    on older multio rather than producing spurious failures.
    """

    # provenance (filled by the loader) ----------------------------------
    source: str = ""
    id: str = ""

    @model_validator(mode="after")
    def _validate_shape(self) -> "Case":
        if (self.step is None) == (self.steps is None):
            raise ValueError(f"case {self.name!r}: set exactly one of 'step' or 'steps'")
        if self.steps is not None and self.expect_sequence is None:
            raise ValueError(f"case {self.name!r}: 'steps' requires 'expect_sequence'")
        if self.step is not None and self.expect_sequence is not None:
            raise ValueError(f"case {self.name!r}: 'expect_sequence' requires 'steps'")
        if self.steps is not None and len(self.steps) != len(self.expect_sequence or []):
            raise ValueError(f"case {self.name!r}: 'steps' and 'expect_sequence' length mismatch")
        return self

    @property
    def step_list(self) -> list[int | str]:
        """Normalise to a list of steps regardless of single/sequence form."""
        return self.steps if self.steps is not None else [self.step]  # type: ignore[list-item]

    def to_worker_dict(self) -> dict[str, Any]:
        """Serialise to the plain-dict shape consumed by ``_worker``.

        The worker runs in a separate interpreter and only needs the encode
        inputs (not the expectations), so we hand it a minimal JSON-friendly
        dict with ``user`` re-aliased to the plugin's ``class`` key.
        """
        name, spec = self.variable.to_transform_dict()
        return {
            "grid": self.grid,
            "timestep": self.timestep,
            "accumulate_from_start": self.accumulate_from_start,
            "cached": self.cached,
            "user": self.user.to_plugin_kwargs(),
            "reference": {"date": self.reference.date, "time": self.reference.time},
            "variable": {"name": name, **spec},
            "step": self.step,
            "steps": self.steps,
            "field": self.field.model_dump(),
        }


class CaseFile(BaseModel):
    """Top-level YAML document: shared ``defaults`` + a list of ``cases``."""

    model_config = ConfigDict(extra="forbid")

    defaults: dict[str, Any] = Field(default_factory=dict)
    cases: list[dict[str, Any]]

    def resolved(self, source: str) -> list[Case]:
        """Merge defaults into each raw case and validate to :class:`Case`."""
        out: list[Case] = []
        for raw in self.cases:
            merged = _deep_merge(self.defaults, raw)
            merged["source"] = source
            merged["id"] = f"{source.removesuffix('.yaml')}::{raw['name']}"
            out.append(Case.model_validate(merged))
        return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# Keys that mark a metadata key as "checkable string" vs numeric handled by the
# test comparison logic; exported for clarity.
StatProcessing = Literal["instant", "accumulation", "maximum", "minimum"]
