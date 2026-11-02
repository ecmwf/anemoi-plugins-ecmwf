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
        per output point. When it does not (e.g. stretched / cutout grids, whose
        lookup grid and area describe only one of the sources), or when MIR
        cannot build the grid or area, the template is rebuilt as an
        unstructured grid from the state's latitudes/longitudes (area dropped).
        Without a grid in the lookup the state coordinates are used directly.
        """
        _lookup = lookup.copy()

        grid = _lookup.pop("grid", None)
        area = _lookup.pop("area", None)

        if isinstance(grid, (int, float)) and not isinstance(grid, bool):
            # anemoi-inference collapses a [0.5, 0.5] grid to 0.5, which MIR does not accept
            grid = f"{grid}/{grid}"

        coordinates = self._coordinates(kwargs.get("state"))

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

        if grid is not None:
            try:
                template = MIRRegrid(grid=grid, area=area).forward(base_template)[0]
            except Exception as e:  # MIR raises generic RuntimeError / SpecError
                if coordinates is None:
                    raise
                reason = f"MIR cannot build grid {grid!r} with area {area!r} ({e})"
            else:
                npoints = template.metadata("numberOfDataPoints")
                if coordinates is None or npoints == len(coordinates[0]):
                    return template
                reason = (
                    f"grid {grid!r} with area {area!r} has {npoints} points but the output has {len(coordinates[0])}"
                )

            LOG.info("Using the output latitudes/longitudes as an unstructured grid: %s", reason)

        # Errors from MIR on the unstructured grid are deliberately not caught.
        latitudes, longitudes = coordinates
        unstructured = {"latitudes": latitudes.tolist(), "longitudes": longitudes.tolist()}
        template = MIRRegrid(grid=unstructured).forward(base_template)[0]

        npoints = template.metadata("numberOfDataPoints")
        if npoints != len(latitudes):
            raise ValueError(
                f"Template for {variable} has {npoints} points, but the output state has {len(latitudes)} points"
            )
        return template

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
