# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd
from anemoi.transform.filter import Filter

from .backend import GridSpec
from .backend import mir_regrid

LOG = logging.getLogger(__name__)


class MIRRegrid(Filter):
    """Regrid fields to a target grid and optional area using MIR.

    Parameters
    ----------
    grid : GridSpec
        The target grid specification. Can be a grid string (e.g. "O32"),
        a list/tuple of increments, or a dict of coordinate lists.
    area : str | list[float] | tuple[float, ...] | None, optional
        The target area for regridding, by default None.
    packing : str, optional
        The packing method to use, by default "ccsds".
    accuracy : int, optional
        The accuracy for regridding, by default 16.
    method : str, optional
        The regridding method to use, by default "grib".
        Can be "array" to use MIR's array-based regridding, or "grib" to use GRIB-based regridding.
    """

    def __init__(
        self,
        *,
        grid: GridSpec,
        area: str | list[float] | tuple[float, ...] | None = None,
        packing: str = "ccsds",
        accuracy: int = 16,
        method: str = "grib",
    ) -> None:
        self._grid = grid
        self._area = area

        self._packing = packing
        self._accuracy = accuracy
        self._method = method

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Regrid the input fields to the target grid.

        Parameters
        ----------
        fields : ekd.FieldList
            The input fields to regrid.

        Returns
        -------
        ekd.FieldList
            The regridded fields.
        """
        return mir_regrid(
            fields,
            self._grid,
            self._area,
            packing=self._packing,
            accuracy=self._accuracy,
            method=self._method,
        )

    def __repr__(self) -> str:
        return f"MIRRegrid(grid={self._grid!r}, area={self._area!r})"
