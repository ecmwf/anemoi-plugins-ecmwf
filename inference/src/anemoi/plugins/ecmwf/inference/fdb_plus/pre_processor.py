# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import time
from contextlib import contextmanager
from typing import Any

import earthkit.data as ekd
from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.processor import Processor
from anemoi.inference.types import DataRequest
from anemoi.inference.types import State
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid
from anemoi.plugins.ecmwf.transform.spectral import ShToGg
from anemoi.plugins.ecmwf.transform.spectral import VordivToUV

LOG = logging.getLogger(__name__)


@contextmanager
def profile(name: str):
    start = time.time()
    yield
    end = time.time()
    LOG.info(f"{name} took {end - start:.3f} seconds")


class FDBPlusPreProcessor(Processor):
    """Enhanced FDB pre-processor that handles optimised vorticity/divergence to u/v conversion and regridding of other fields."""

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        target_grid: str,
        spectral_backend: str | None = None,
    ) -> None:
        super().__init__(context, metadata)
        self.target_grid = target_grid

        self.regrid_processor = MIRRegrid(grid=self.target_grid, packing="simple", accuracy=16, method="array")
        self.vordiv_to_uv_processor = VordivToUV(transform_grid=self.target_grid, backend=spectral_backend)
        self.sh_to_gg_processor = ShToGg(transform_grid=self.target_grid, backend=spectral_backend)

    def process(self, state: State) -> Any:
        fields = state["fields"]

        with profile("Partitioning fields"):
            vor_div = []
            spectral_scalar = []
            gridpoint = []

            for f in fields:
                grid_type = f.metadata("gridType")
                if grid_type == "sh":
                    param = f.metadata("param")
                    if param in ("vo", "d"):
                        vor_div.append(f)
                    else:
                        spectral_scalar.append(f)
                else:
                    gridpoint.append(f)

            # Convert lists to FieldLists for downstream processors
            vor_div_fields = ekd.FieldList.from_fields(vor_div)
            spectral_fields = ekd.FieldList.from_fields(spectral_scalar)
            remaining_fields = ekd.FieldList.from_fields(gridpoint)

        # Convert vorticity/divergence to u/v fields on target grid directly
        with profile("Vordiv to UV conversion"):
            u_v_fields = self.vordiv_to_uv_processor.forward(vor_div_fields)

        # Transform all other spectral (spherical-harmonics) fields to the
        # target grid directly.
        with profile("Spectral to gridpoint conversion"):
            spectral_gg_fields = self.sh_to_gg_processor.forward(spectral_fields)

        # Regrid remaining (grid-point) fields to target grid
        with profile("Regridding remaining fields"):
            regridded_remaining_fields = self.regrid_processor.forward(remaining_fields)

        # combine u/v, spectral scalar and regridded remaining fields
        combined_fields = u_v_fields + spectral_gg_fields + regridded_remaining_fields
        state["fields"] = combined_fields
        state["latitudes"] = next(iter(regridded_remaining_fields)).metadata().geography.latitudes()
        state["longitudes"] = next(iter(regridded_remaining_fields)).metadata().geography.longitudes()

        return state

    def patch_data_request(self, data_request: DataRequest) -> DataRequest:
        data_request = self.vordiv_to_uv_processor.patch_data_request(data_request)
        return data_request
