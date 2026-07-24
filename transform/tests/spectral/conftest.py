# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import pytest
from anemoi.plugins.ecmwf.transform.spectral.backends import backend_registry


def _available_backend_names():
    """Return list of backend names that are currently available."""
    available = []
    for name, cls in backend_registry.factories.items():
        ok, _ = cls.available()
        if ok:
            available.append(name)
    return available


def any_spectral_backend_available():
    """Check if any spectral backend is available."""
    return len(_available_backend_names()) > 0


@pytest.fixture(params=_available_backend_names() or ["none_available"], ids=lambda x: x)
def backend_name(request):
    """Parametrised fixture yielding each available backend name."""
    if request.param == "none_available":
        pytest.skip("No spectral backend available")
    return request.param
