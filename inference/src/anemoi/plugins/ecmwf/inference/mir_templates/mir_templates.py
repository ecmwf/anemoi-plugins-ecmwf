# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import base64
import io
import logging
import os
import zlib
from typing import Any

import earthkit.data as ekd
import mir
from anemoi.inference.grib.templates import IndexTemplateProvider
from anemoi.inference.grib.templates import TemplateProvider

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

    def _regrid_with_mir(self, base_template: bytes, grid: str, area: str | None = None) -> bytes:
        """Regrid the base template using mir.

        Parameters
        ----------
        base_template : bytes
            Base grib handle template as bytes to regrid.
        grid : str
            Target grid for regridding.
        area : Optional[str], optional
            Target area for regridding, by default None

        Returns
        -------
        bytes
            Grib handle in bytes regridded.
        """

        mir_input = mir.GribMemoryInput(base_template)
        job_args = {"grid": grid}
        if area:
            job_args["area"] = area

        job = mir.Job(**job_args)
        buffer = io.BytesIO()

        job.execute(mir_input, buffer)

        return buffer.getvalue()

    def template(self, variable: str, lookup: dict[str, Any]) -> ekd.Field:
        """Get the template for the given variable and lookup.

        Parameters
        ----------
        variable : str
            The variable to get the template for.
        lookup : Dict[str, Any]
            The lookup dictionary.

        Returns
        -------
        ekd.Field
            The template field.
        """
        _lookup = lookup.copy()

        grid = _lookup.pop("grid")
        area = _lookup.pop("area", None)

        if isinstance(grid, (list, tuple)):
            grid = "/".join(map(str, grid))
        if isinstance(area, (list, tuple)):
            area = "/".join(map(str, area))

        if isinstance(grid, (int, float)):
            grid = f"{grid}/{grid}"  # Convert single value to a grid string, introduced by choice in anemoi-inference.

        base_template = self._base_template_provider.template(variable, lookup)
        if base_template is None:
            raise ValueError(f"Base template not found for variable {variable} with lookup {lookup}")

        regridded_template = self._regrid_with_mir(base_template, str(grid), area)

        if len(regridded_template) == 0:
            raise ValueError(f"Regridded template is empty for variable {variable} with lookup {lookup}")

        return ekd.from_source("memory", regridded_template)[0]  # type: ignore
