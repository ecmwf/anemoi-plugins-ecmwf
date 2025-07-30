# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import TYPE_CHECKING

import earthkit.data as ekd
import tqdm
from anemoi.inference.context import Context
from anemoi.inference.processor import Processor

if TYPE_CHECKING:
    from earthkit.data.readers.grib.codes import GribField

LOG = logging.getLogger(__name__)


def _mir_regrid(field: "GribField", grid, area) -> "GribField":
    import io

    import mir
    from earthkit.data import create_encoder

    encoder = create_encoder("grib")
    message = encoder.encode(field).to_bytes()

    mir_input = mir.GribMemoryInput(message)  # type: ignore
    job_args = {"grid": grid}
    if area:
        job_args["area"] = area

    job = mir.Job(**job_args)  # type: ignore
    buffer = io.BytesIO()

    job.execute(mir_input, buffer)

    return ekd.from_source("memory", buffer.getvalue())[0]  # type: ignore


def regrid(fields: ekd.FieldList, grid, area) -> ekd.FieldList:
    """Regrid a list of fields to a specified grid and area.

    TO BE REPLACED WITH EARTHKIT-REGRID
    """
    if isinstance(grid, (list, tuple)):
        grid = "/".join(map(str, grid))
    if isinstance(area, (list, tuple)):
        area = "/".join(map(str, area))

    result = list(map(lambda f: _mir_regrid(f, grid, area), tqdm.tqdm(fields, desc="Regridding fields")))  # type: ignore
    return ekd.FieldList.from_fields(result)


class RegridPreprocessor(Processor):
    """Regrid an input fieldlist.

    Can only be used when working with grib, from any of the earthkit data sources.
    i.e. mars, cds, opendata, grib files, etc.
    """

    def __init__(self, context: Context, grid: str | list[float], area: str | list[float] | None = None) -> None:
        """Initialize the Regridding processor.

        Parameters
        ----------
        context : Context
            The context in which the processor operates.
        grid : str | list[float]
            The target grid for regridding.
        area : str | list[float] | None, optional
            The target area for regridding, by default None
        """
        super().__init__(context)
        self._grid = grid
        self._area = area

    def process(self, fields: ekd.FieldList) -> ekd.FieldList:  # type: ignore
        """Process the fields to replace NaNs with the mean value.

        Parameters
        ----------
        fields : list
            List of fields to process.

        Returns
        -------
        list
            List of processed fields with NaNs replaced by the mean value.
        """
        return regrid(fields, self._grid, self._area)
