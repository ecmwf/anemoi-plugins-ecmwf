# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import base64
import logging
import os
import zlib
from typing import Any

import earthkit.data as ekd
import numpy as np
from anemoi.inference.grib.templates import IndexTemplateProvider
from anemoi.inference.grib.templates import TemplateProvider
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid
from anemoi.plugins.ecmwf.transform.regrid.backend import _make_mir_grid
from anemoi.plugins.ecmwf.transform.regrid.backend import grid_repr

LOG = logging.getLogger(__name__)


class BaseTemplateProvider(IndexTemplateProvider):
    def load_template(self, grib: str, lookup: dict[str, Any]) -> bytes:  # type: ignore
        return zlib.decompress(base64.b64decode(grib))


class MirTemplatesProvider(TemplateProvider):
    """Template provider using mir to make new grid templates."""

    def __init__(self, manager: Any, path: str | None = None) -> None:
        """Initialise the MirTemplatesProvider instance.

        Parameters
        ----------
        manager : Any
            The manager instance.
        path : str
            The path to the base handles file.
            Expected to be in the templates yaml format, and
            have no grid / area keys.
        """
        self.manager = manager

        if path is None:
            path = os.path.join(os.path.dirname(__file__), "base_handles.yaml")

        self._base_template_provider = BaseTemplateProvider(manager, path)

    def template(self, variable: str, lookup: dict[str, Any], **kwargs) -> ekd.Field | None:
        """Get the template for the given variable and lookup.

        Parameters
        ----------
        variable : str
            The variable to get the template for.
        lookup : Dict[str, Any]
            The lookup dictionary.
        kwargs
            Extra arguments for specific template providers.
            ``state`` (the output state, with ``latitudes`` and ``longitudes``)
            is used to build templates for unstructured grids.

        Returns
        -------
        ekd.Field | None
            The template field, or None if the grid cannot be determined.

        Notes
        -----
        The grid from the lookup is used as is when it has a ``area``, or when
        it matches the number of output points. Otherwise (no grid, a grid
        MIR cannot build, or a size mismatch, e.g. stretched / cutout grids)
        the output latitudes/longitudes are used as an unstructured MIR grid.
        """
        _lookup = lookup.copy()

        grid = _lookup.pop("grid", None)
        area = _lookup.pop("area", None)

        state = kwargs.get("state")
        unstructured = self._unstructured_grid(state)

        if grid is None:
            if unstructured is None:
                LOG.warning(
                    "No grid in the template lookup and no latitudes/longitudes in the state, "
                    "cannot build a template for %s.",
                    variable,
                )
                return None
            grid, area = unstructured, None
        elif unstructured is not None and area is None:
            regridder = MIRRegrid(grid=grid, area=None)
            reason = self._grid_mismatch(regridder.grid, len(unstructured["latitudes"]))
            if reason is not None:
                LOG.info("Using the output latitudes/longitudes as an unstructured grid: %s", reason)
                grid = unstructured

        base_template = ekd.from_source("memory", self._base_template_provider.template(variable, lookup))
        if base_template is None:
            raise ValueError(f"Base template not found for variable {variable} with lookup {lookup}")

        # Errors from MIR (e.g. on an unstructured grid) are deliberately not caught.
        regridder = MIRRegrid(grid=grid, area=area)
        return regridder.forward(base_template)[0]

    @staticmethod
    def _unstructured_grid(state: dict[str, Any] | None) -> dict[str, list[float]] | None:
        """Build an unstructured MIR grid specification from the state's latitudes/longitudes."""
        if not state:
            return None

        latitudes = state.get("latitudes")
        longitudes = state.get("longitudes")
        if latitudes is None or longitudes is None:
            return None

        latitudes = np.asarray(latitudes, dtype=float).ravel()
        longitudes = np.asarray(longitudes, dtype=float).ravel()
        if latitudes.size == 0 or latitudes.size != longitudes.size:
            return None

        return {"latitudes": latitudes.tolist(), "longitudes": longitudes.tolist()}

    @staticmethod
    def _grid_mismatch(grid: Any, npoints: int) -> str | None:
        """Return why ``grid`` cannot describe ``npoints`` output points, or None if it can."""
        try:
            size = _make_mir_grid(grid).size()
        except Exception as e:  # MIR raises generic RuntimeError / SpecError
            return f"MIR cannot build grid {grid_repr(grid)!r} ({e})"

        if size != npoints:
            return f"grid {grid_repr(grid)!r} has {size} points but the output has {npoints}"
        return None
