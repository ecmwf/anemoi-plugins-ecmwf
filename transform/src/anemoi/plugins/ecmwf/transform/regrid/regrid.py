# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os

import earthkit.data as ekd
from anemoi.transform.filter import Filter

from .backend import GridSpec
from .backend import canonical_lambert_grid
from .backend import check_lambert_area
from .backend import grid_repr
from .backend import is_lambert_grid
from .backend import mir_regrid
from .named import KNOWN_GRIDS
from .named import NamedRegrid

LOG = logging.getLogger(__name__)


class MIRRegrid(Filter):
    """Regrid fields to a target grid and optional area using MIR.

    Parameters
    ----------
    grid : GridSpec
        The target grid specification. Can be a grid string (e.g. "O32"),
        a list/tuple of increments, a dict of ``latitudes`` and ``longitudes``
        lists, or a Lambert grid: a dict of GRIB keys with ``gridType: lambert``
        (see :func:`~anemoi.plugins.ecmwf.transform.regrid.backend.canonical_lambert_grid`).
        A Lambert grid is validated and stored in its canonical form; the output fields
        report it as their MARS grid, and no MARS area.
    area : str | list[float] | tuple[float, ...] | None, optional
        The target area for regridding, by default None. A ``ValueError`` is raised for an
        area with a Lambert grid, which defines its own domain and cannot be cropped by MIR.
    packing : str, optional
        The packing method to use, by default "ccsds".
    accuracy : int, optional
        The accuracy for regridding, by default 16.
    """

    def __init__(
        self,
        *,
        grid: GridSpec,
        area: str | list[float] | tuple[float, ...] | None = None,
        packing: str = "ccsds",
        accuracy: int = 16,
    ) -> None:

        if is_lambert_grid(grid):
            grid = canonical_lambert_grid(grid)  # type: ignore[arg-type]
            check_lambert_area(grid, area)
        elif isinstance(grid, str) and grid.lower() in KNOWN_GRIDS:
            named_regrid = NamedRegrid(grid)
            grid = named_regrid.gridspec["grid"]

        self._grid = grid
        self._area = area

        self._packing = packing
        self._accuracy = accuracy

    @property
    def grid(self) -> GridSpec:
        return self._grid

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
        if is_lambert_grid(self._grid) and os.environ.get("ECCODES_ECKIT_GEO") == "1":
            LOG.warning(
                "Regridding to a Lambert grid with ECCODES_ECKIT_GEO=1: ecCodes cannot build the geography of "
                "Lambert grids, so reading the grid points of the output fails. "
                "Set ECCODES_ECKIT_GEO=0 in the environment before starting Python.",
            )

        return mir_regrid(
            fields,
            self._grid,
            self._area,
            packing=self._packing,
            accuracy=self._accuracy,
        )

    def __repr__(self) -> str:
        return f"MIRRegrid(grid={grid_repr(self._grid)!r}, area={self._area!r})"
