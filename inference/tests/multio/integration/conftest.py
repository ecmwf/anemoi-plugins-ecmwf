# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared fixtures / helpers for the multio -> GRIB integration suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from .schema import Case
from .schema import CaseFile

CASES_DIR = Path(__file__).parent / "cases"


def load_cases() -> list[Case]:
    """Load, default-merge and validate every case from every ``cases/*.yaml``.

    Each file is parsed as a :class:`CaseFile`; its ``defaults`` are deep-merged
    into each case which is then validated as a :class:`Case`. A malformed case
    raises at collection time with a pydantic error pointing at the field.
    """
    resolved: list[Case] = []
    for path in sorted(CASES_DIR.glob("*.yaml")):
        doc = CaseFile.model_validate(yaml.safe_load(path.read_text()))
        resolved.extend(doc.resolved(path.name))
    return resolved


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: real multio->GRIB encode/roundtrip tests (require pymultio + earthkit)",
    )
