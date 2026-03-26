# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plugin that subtracts precomputed tendencies.

At every autoregressive step the model predicts the *full* next state.
This processor removes the tendency so that only the
anomaly is fed back into the next step.
"""

import logging
from typing import Any

import earthkit.data as ekd
import numpy as np
import torch
from anemoi.inference.processor import Processor
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)


class SubtractTendencyPlugin(Processor):
    """Subtract precomputed tendencies from model output each step.

    This is a **mid-processor**: it runs inside the autoregressive loop
    *after* the model prediction and *before* the output is fed back as
    input for the next step.

    Parameters
    ----------
    context : Context
        The inference runner context.
    tend_pl_path : str
        Path to the pressure-level tendency GRIB file.
    tend_sfc_path : str
        Path to the surface tendency GRIB file.
    param_pl : list[str]
        Pressure-level parameter short names (e.g. ``["z", "q", "t", "u", "v", "w"]``).
    level_pl : list[int]
        Pressure levels (e.g. ``[1000, 925, 850, ...]``).
    param_sfc : list[str]
        Surface parameter short names (e.g. ``["msl", "10u", "10v", "2t"]``).

    Example
    -------
    .. code-block:: yaml

        mid_processors:
          - subtract_tendency:
              tend_pl_path: "/path/to/tend_pl_JAS_o96.grib"
              tend_sfc_path: "/path/to/tend_sfc_JAS_o96.grib"
              param_pl: ["z", "q", "t", "u", "v", "w"]
              level_pl: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
              param_sfc: ["msl", "10u", "10v", "2t"]
    """

    def __init__(
        self,
        context: Any,
        tend_pl_path: str,
        tend_sfc_path: str,
        param_pl: list[str],
        level_pl: list[int],
        param_sfc: list[str],
    ) -> None:
        super().__init__(context)

        self._tend_pl_path = tend_pl_path
        self._tend_sfc_path = tend_sfc_path
        self._param_pl = param_pl
        self._level_pl = level_pl
        self._param_sfc = param_sfc

        # Load tendency GRIB data and build lookup: variable_name -> 1-D numpy array
        self._tendency_np = self._load_tendencies()
        LOG.info(
            "SubtractTendency: loaded %d tendency fields from %s and %s",
            len(self._tendency_np),
            tend_pl_path,
            tend_sfc_path,
        )

        # Lazily populated torch tensor cache (device/dtype matched on first call)
        self._tendency_torch: dict[str, Any] = {}

    def _load_tendencies(self) -> dict[str, np.ndarray]:
        """Load tendency GRIB files and return {variable_name: 1-D array}."""
        tendency: dict[str, np.ndarray] = {}

        # --- Pressure levels ---
        tend_pl = (
            ekd.from_source("file", self._tend_pl_path)
            .order_by(param=self._param_pl, level=self._level_pl)
        )
        tend_pl_values = tend_pl.values  # shape (n_fields, n_gridpoints)

        idx = 0
        for param in self._param_pl:
            for level in self._level_pl:
                name = f"{param}_{level}"
                tendency[name] = tend_pl_values[idx].astype(np.float32)
                idx += 1

        # --- Surface ---
        tend_sfc = (
            ekd.from_source("file", self._tend_sfc_path)
            .order_by(param=self._param_sfc)
        )
        tend_sfc_values = tend_sfc.values  # shape (n_fields, n_gridpoints)

        for i, param in enumerate(self._param_sfc):
            tendency[param] = tend_sfc_values[i].astype(np.float32)

        return tendency

    def _get_tendency_tensor(self, name: str, reference) -> "torch.Tensor":
        """Return the tendency for *name* as a torch tensor on the same device/dtype."""
        import torch

        if name not in self._tendency_torch:
            arr = self._tendency_np[name]
            self._tendency_torch[name] = torch.from_numpy(arr).to(
                device=reference.device, dtype=reference.dtype
            )
        return self._tendency_torch[name]

    def process(self, state: State) -> State:
        """Subtract the tendency from each matching field in the state."""
        state = state.copy()

        for name, tend_np in self._tendency_np.items():
            if name not in state["fields"]:
                continue
            field = state["fields"][name]
            tend = self._get_tendency_tensor(name, field)
            state["fields"][name] = field - tend

        return state

    def __repr__(self) -> str:
        return (
            f"SubtractTendencyPlugin("
            f"pl={self._tend_pl_path!r}, sfc={self._tend_sfc_path!r}, "
            f"n_fields={len(self._tendency_np)})"
        )
