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
from typing import Dict
from typing import List
from typing import Optional

import earthkit.data as ekd
import mir
import yaml
from anemoi.inference.grib.templates import TemplateProvider

LOG = logging.getLogger(__name__)


class MirTemplatesProvider(TemplateProvider):
    """Template provider using mir to make new grid templates."""

    def __init__(self, manager: Any, path: Optional[str] = None) -> None:
        """Initialise the MirTemplatesProvider instance.

        Parameters
        ----------
        manager : Any
            The manager instance.
        path : str
            The path to the base handles file.
        """
        self.manager = manager

        if path is None:
            path = os.path.join(os.path.dirname(__file__), "base_handles.yaml")

        with open(path) as f:
            self.templates = yaml.safe_load(f)

        if not isinstance(self.templates, list):
            raise ValueError("Invalid templates.yaml, must be a list")

        # TODO: use pydantic
        for template in self.templates:
            if not isinstance(template, list):
                raise ValueError(f"Invalid template in templates.yaml, must be a list: {template}")
            if len(template) != 2:
                raise ValueError(f"Invalid template in templates.yaml, must have exactly 2 elements: {template}")

            match, grib = template
            if not isinstance(match, dict):
                raise ValueError(f"Invalid match in templates.yaml, must be a dict: {match}")

            if not isinstance(grib, str):
                raise ValueError(f"Invalid grib in templates.yaml, must be a string: {grib}")

    def _base_template(self, variable: str, lookup: Dict[str, Any]) -> Optional[bytes]:
        """Get the base template for the given variable and lookup.

        Parameters
        ----------
        variable : str
            The variable to get the template for.
        lookup : Dict[str, Any]
            The lookup dictionary.

        Returns
        -------
        bytes
            The template handle bytes, or None if not found.
        """

        def _as_list(value: Any) -> List[Any]:
            if not isinstance(value, list):
                return [value]
            return value

        for template in self.templates:
            match, grib = template
            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug("%s", [(lookup.get(k), _as_list(v)) for k, v in match.items()])

            if all(lookup.get(k) in _as_list(v) for k, v in match.items()):
                return zlib.decompress(base64.b64decode(grib))
        return None

    def _regrid_with_mir(self, base_template: bytes, grid: str, area: Optional[str] = None) -> bytes:
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

    def template(self, variable: str, lookup: Dict[str, Any]) -> ekd.Field:
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

        base_template = self._base_template(variable, _lookup)
        if base_template is None:
            raise ValueError(f"Base template not found for variable {variable} with lookup {lookup}")

        regridded_template = self._regrid_with_mir(base_template, grid, area)

        if len(regridded_template) == 0:
            raise ValueError(f"Regridded template is empty for variable {variable} with lookup {lookup}")

        return ekd.from_source("memory", regridded_template)[0]
