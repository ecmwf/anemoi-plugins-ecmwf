# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter

from ..utils import fields_to_numpy_parallel
from .backends import make_backend

LOG = logging.getLogger(__name__)


class ShToGg(Filter):
    """Transform spectral scalar fields (spherical harmonics) to grid-point.

    Companion to :class:`VordivToUV` for scalar fields (e.g. temperature,
    vertical velocity), using the same spectral-transform backends.
    All input fields must be spectral and share the same truncation.
    """

    def __init__(
        self,
        *,
        target_grid: str | None = None,
        transform_grid: str | None = None,
    ):
        """Initialise the ShToGg filter.

        If `transform_grid` is specified, spectral fields are truncated to
        this grid's resolution before the transform (cheaper Legendre/FLT
        setup at the expense of small-scale detail), and the output fields
        are on that grid. Otherwise `target_grid` is used, and if neither
        is given the corresponding octahedral grid is derived from the
        spectral size of the input fields.

        Parameters
        ----------
        target_grid: str, optional
            Target grid for the output fields. If None, derived from the
            spectral size of the input fields.
        transform_grid: str, optional
            Pre-truncate spectral fields to this grid's resolution before
            transforming. Also determines the output grid.
        """
        self.target_grid = target_grid
        self.transform_grid = transform_grid

    def _resolve_grid(self, nspec: int) -> tuple[str, int]:
        from .utils import grid_to_trunc
        from .utils import trunc_from_nspec

        if self.transform_grid is not None:
            return self.transform_grid, grid_to_trunc(self.transform_grid)

        if self.target_grid is not None:
            return self.target_grid, grid_to_trunc(self.target_grid)

        trunc = trunc_from_nspec(nspec)
        return f"O{trunc + 1}", trunc

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        if len(fields) == 0:
            return fields

        from .utils import truncate_spectral

        data = fields_to_numpy_parallel(fields)  # (nfields, nspec)
        grid, trunc = self._resolve_grid(data.shape[-1])

        data = truncate_spectral(data, trunc)
        LOG.debug("sh -> gg: transforming %d fields at T%d (%s)", len(fields), trunc, grid)

        backend = make_backend(grid, trunc)
        gridpoint = backend.sh_to_gg(data)

        return new_fieldlist_from_list([new_field_from_numpy(gridpoint[i], template=f) for i, f in enumerate(fields)])

    def backward(self, fields: ekd.FieldList) -> ekd.FieldList:
        raise NotImplementedError("ShToGg does not implement the backward (gg -> sh) transform")
