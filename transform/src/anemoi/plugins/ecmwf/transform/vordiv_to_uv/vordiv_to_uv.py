# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections.abc import Iterator

import earthkit.data as ekd
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter
from anemoi.transform.filters.fields.matching import MatchingSpec

from .backends import CalculationBackend
from .backends import get_backend

LOG = logging.getLogger(__name__)


class VordivToUV(MatchingFieldsFilter):
    MATCHING = MatchingSpec(
        select="param",
        forward=("vorticity", "divergence"),
        backward=("u_component_of_wind", "v_component_of_wind"),
        vertical=True,
    )

    def __init__(
        self,
        *,
        vorticity: str = "vo",
        divergence: str = "d",
        u_component_of_wind: str = "u",
        v_component_of_wind: str = "v",
        target_grid: str | None = None,
        spectral_grid: str | None = None,
        transform_grid: str | None = None,
    ):
        """Initialise the VordivToUV filter.

        Compute the u and v components of wind from vorticity and divergence.

        If `target_grid` is specified, the output fields will be on that grid.
        If not specified, the corresponding octahedral grid will be derived from
        the spectral size of the input fields.

        If `spectral_grid` is specified, it is used for the backward transform
        (u/v -> vor/div) to determine the spectral truncation and output size.
        If not specified, the backward transform will derive it from the
        grid-point data using `target_grid`.

        If `transform_grid` is specified, spectral fields are truncated to
        this grid's resolution before the transform.  This reduces the cost
        of the Legendre/FLT setup at the expense of discarding small-scale
        detail.  For example, ``transform_grid="O320"`` truncates T1279
        input to T319 before transforming, which is much cheaper.

        Parameters
        ----------
        vorticity: str, optional
            Name of the vorticity parameter, by default "vo".
        divergence: str, optional
            Name of the divergence parameter, by default "d".
        u_component_of_wind: str, optional
            Name of the u component of wind parameter, by default "u".
        v_component_of_wind: str, optional
            Name of the v component of wind parameter, by default "v".
        target_grid: str, optional
            Target grid for the forward transform output (vor/div -> u/v).
            If None, derived from the spectral size of the input fields.
        spectral_grid: str, optional
            Target spectral grid for the backward transform output
            (u/v -> vor/div).  If None, derived from `target_grid`.
        transform_grid: str, optional
            Pre-truncate spectral fields to this grid's resolution before
            transforming.  Reduces setup cost for high-resolution inputs.
            Also determines the output grid for the forward transform.
        """
        self.vorticity = vorticity
        self.divergence = divergence
        self.u_component_of_wind = u_component_of_wind
        self.v_component_of_wind = v_component_of_wind

        self.target_grid = target_grid
        self.spectral_grid = spectral_grid
        self.transform_grid = transform_grid

        super().__init__()

    def _resolve_forward_grid(self, vorticity: ekd.Field) -> tuple[str, int]:
        """Resolve the grid and truncation for the forward transform."""
        from .utils import grid_to_trunc

        # transform_grid takes precedence: it sets both truncation and output grid
        if self.transform_grid is not None:
            return self.transform_grid, grid_to_trunc(self.transform_grid)

        if self.target_grid is not None:
            return self.target_grid, grid_to_trunc(self.target_grid)

        from .utils import trunc_from_nspec

        nspec = vorticity.to_numpy().shape[-1]
        trunc = trunc_from_nspec(nspec)
        return f"O{trunc + 1}", trunc

    def _resolve_backward_grid(self) -> tuple[str, int]:
        """Resolve the grid and truncation for the backward transform."""
        from .utils import grid_to_trunc

        if self.spectral_grid is not None:
            return self.spectral_grid, grid_to_trunc(self.spectral_grid)

        if self.transform_grid is not None:
            return self.transform_grid, grid_to_trunc(self.transform_grid)

        raise ValueError("spectral_grid or transform_grid must be set for " "u/v -> vor/div backward transform")

    def _make_backend(self, grid: str, trunc: int) -> CalculationBackend:
        """Create a backend for the given grid and truncation."""
        import warnings

        from .utils import grid_to_pl

        backend_class = get_backend()

        try:
            kloen = grid_to_pl(grid)
        except ValueError:
            fallback_grid = f"O{trunc + 1}"
            warnings.warn(
                f"Grid '{grid}' pl not available in lookup. "
                f"Falling back to '{fallback_grid}' (same truncation T{trunc}).",
                stacklevel=3,
            )
            kloen = grid_to_pl(fallback_grid)

        return backend_class(kloen, trunc)

    def forward_transform(
        self, vorticity: ekd.FieldList, divergence: ekd.FieldList
    ) -> Iterator[ekd.Field]:  # type: ignore[reportIncompatibleMethodOverride]
        grid, trunc = self._resolve_forward_grid(vorticity)
        backend = self._make_backend(grid, trunc)

        vor_np = vorticity.to_numpy()
        div_np = divergence.to_numpy()

        # Pre-truncate spectral fields if transform_grid sets a lower truncation
        if self.transform_grid is not None:
            from .utils import truncate_spectral

            vor_np = truncate_spectral(vor_np, trunc)
            div_np = truncate_spectral(div_np, trunc)
            LOG.debug(
                "Truncated spectral fields to T%d (%s) before transform",
                trunc,
                self.transform_grid,
            )

        u, v = backend.vordiv_to_uv(vor_np, div_np)

        for i, f in enumerate(vorticity):  # type: ignore[reportArgumentType]
            yield self.new_field_from_numpy(u[i], template=f, param=self.u_component_of_wind)
            yield self.new_field_from_numpy(v[i], template=f, param=self.v_component_of_wind)

    def backward_transform(
        self, u_component_of_wind: ekd.Field, v_component_of_wind: ekd.Field
    ) -> Iterator[ekd.Field]:  # type: ignore[reportIncompatibleMethodOverride]
        grid, trunc = self._resolve_backward_grid()
        backend = self._make_backend(grid, trunc)
        vor, div = backend.uv_to_vordiv(u_component_of_wind.to_numpy(), v_component_of_wind.to_numpy())

        for i, f in enumerate(u_component_of_wind):  # type: ignore[reportArgumentType]
            yield self.new_field_from_numpy(vor[i], template=f, param=self.vorticity)
            yield self.new_field_from_numpy(div[i], template=f, param=self.divergence)

    def patch_data_request(self, data_request: dict) -> dict:
        if any(
            param in data_request.get("param", []) for param in [self.u_component_of_wind, self.v_component_of_wind]
        ):
            param: list[str] = data_request.get("param", [])
            param.remove(self.u_component_of_wind)
            param.remove(self.v_component_of_wind)
            param.append(self.vorticity)
            param.append(self.divergence)
            data_request["param"] = param
        return data_request
