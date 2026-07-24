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

    def template(self, variable: str, lookup: dict[str, Any], **kwargs) -> ekd.Field:
        """Get the template for the given variable and lookup.

        Parameters
        ----------
        variable : str
            The variable to get the template for.
        lookup : Dict[str, Any]
            The lookup dictionary.
        kwargs
            Extra arguments for specific template providers.

        Returns
        -------
        ekd.Field
            The template field.
        """
        _lookup = lookup.copy()

        grid = _lookup.pop("grid")
        area = _lookup.pop("area", None)

        base_template = ekd.from_source("memory", self._base_template_provider.template(variable, lookup))
        if base_template is None:
            raise ValueError(f"Base template not found for variable {variable} with lookup {lookup}")

        regridder = MIRRegrid(grid=grid, area=area)
        return regridder.forward(base_template)[0]
