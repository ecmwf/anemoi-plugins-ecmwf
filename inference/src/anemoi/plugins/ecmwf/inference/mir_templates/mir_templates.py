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
        The template is built from the ``grid`` and ``area`` of the lookup.
        If the state has latitudes/longitudes, the template must have one point
        per output point. The size of the lookup grid (and area) is checked up
        front with ``mir.Grid``. When it differs (e.g. stretched / cutout grids,
        whose lookup grid and area describe only one of the sources), the
        state's latitudes/longitudes are used instead, as an unstructured grid
        (area dropped). A grid or area MIR cannot build raises.
        Without a grid in the lookup the state coordinates are used directly.
        The base field is regridded once, to the chosen grid.
        """
        _lookup = lookup.copy()

        grid = _lookup.pop("grid", None)
        area = _lookup.pop("area", None)

        if isinstance(grid, (int, float)) and not isinstance(grid, bool):
            # anemoi-inference collapses a [0.5, 0.5] grid to 0.5, which MIR does not accept
            grid = f"{grid}/{grid}"

        coordinates = self._coordinates(kwargs.get("state"))
        npoints = len(coordinates[0]) if coordinates is not None else None

        if grid is None and coordinates is None:
            LOG.warning(
                "No grid in the template lookup and no latitudes/longitudes in the state, "
                "cannot build a template for %s.",
                variable,
            )
            return None

        base_template = ekd.from_source("memory", self._base_template_provider.template(variable, lookup))
        if base_template is None:
            raise ValueError(f"Base template not found for variable {variable} with lookup {lookup}")

        regrid = None
        if grid is not None:
            # Also resolves named grids to latitudes/longitudes
            lookup_regrid = MIRRegrid(grid=grid, area=area)
            if coordinates is None:
                regrid = lookup_regrid
            else:
                if isinstance(lookup_regrid.grid, dict):
                    # Coordinate lookup grids are not compared (mir.Grid ignores their area):
                    # the state coordinates are the output points.
                    reason = f"grid {grid!r} is given as coordinates"
                else:
                    # A grid or area MIR cannot build raises
                    lookup_npoints = self._grid_size(lookup_regrid.grid, area)
                    if lookup_npoints == npoints:
                        regrid = lookup_regrid
                    else:
                        reason = (
                            f"grid {grid!r} with area {area!r} has {lookup_npoints} points but the output has {npoints}"
                        )

                if regrid is None:
                    LOG.info("Using the output latitudes/longitudes as an unstructured grid: %s", reason)

        if regrid is None:
            latitudes, longitudes = coordinates
            regrid = MIRRegrid(grid={"latitudes": latitudes.tolist(), "longitudes": longitudes.tolist()})

        # Errors from MIR on the final grid are deliberately not caught.
        template = regrid.forward(base_template)[0]

        if npoints is not None:
            template_npoints = template.metadata("numberOfDataPoints")
            if template_npoints != npoints:
                raise ValueError(
                    f"Template for {variable} has {template_npoints} points, but the output state has {npoints} points"
                )
        return template

    @staticmethod
    def _grid_size(grid: Any, area: Any) -> int:
        """Return the number of points of ``grid`` (within ``area``), using ``mir.Grid``.

        Structured grids only (regular or Gaussian grids, with or without an area).
        """
        import mir

        kwargs: dict[str, Any] = {}
        if isinstance(grid, str):
            kwargs["grid"] = grid.upper()
        elif isinstance(grid, (list, tuple)):
            kwargs["grid"] = list(grid)
        else:
            raise ValueError(f"Unsupported grid specification: {grid!r}")

        if area:
            kwargs["area"] = area

        return mir.Grid(**kwargs).size()

    @staticmethod
    def _coordinates(state: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the state's latitudes and longitudes as flat arrays, if usable."""
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

        return latitudes, longitudes
