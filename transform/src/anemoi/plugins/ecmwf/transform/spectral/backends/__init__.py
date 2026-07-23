# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

from anemoi.utils.registry import Registry

from .base import CalculationBackend

LOG = logging.getLogger(__name__)

backend_registry = Registry[CalculationBackend](__name__)


def get_backend(order: list[str] | None = None) -> type[CalculationBackend]:
    """Return the first available backend class.

    Parameters
    ----------
    order : list of str, optional
        Backend names to try, in preference order.  Defaults to all registered.

    Raises
    ------
    RuntimeError
        If no backend is available.

    Returns
    -------
    type[CalculationBackend]
        The first available backend class.
    """
    if order is None:
        order = list(backend_registry.factories.keys())

    error_messages = []
    for name in order:
        cls = backend_registry.lookup(name)
        if cls is None:
            error_messages.append(f"Backend {name} not found")
            continue

        ok, msg = cls.available()
        if ok:
            LOG.debug("Spectral operations: Using backend: %s", name)
            return cls
        error_messages.append(f"{name}: {msg}")

    raise RuntimeError("No available backend found:\n" + "\n".join(error_messages))


def make_backend(grid: str, trunc: int, order: list[str] | None = None) -> CalculationBackend:
    """Create a backend for the given grid and truncation, ordered based on order, or the first available.

    Falls back to the octahedral grid of the same truncation if the grid's
    pl array is not available in the lookup.
    """
    import warnings

    from ..utils import grid_to_pl

    backend_class = get_backend(order)

    try:
        kloen = grid_to_pl(grid)
    except ValueError:
        fallback_grid = f"O{trunc + 1}"
        warnings.warn(
            f"Grid '{grid}' pl not available in lookup. "
            f"Falling back to '{fallback_grid}' (same truncation T{trunc}).",
            stacklevel=3,
        )
        kloen = grid_to_pl(fallback_grid)
        grid = fallback_grid

    return backend_class(kloen, trunc, grid=grid)
