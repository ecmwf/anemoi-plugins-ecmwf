# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Steady-state runner plugin.

Provides a runner that uses the real checkpoint model but freezes dynamic
forcings (e.g. insolation, local solar time) at their initial values for
the entire rollout.  This is useful when combined with mid-processors that
remove the climatological drift each step — for example ``subtract_tendency``
— so that only the anomaly response of a perturbation is retained.

Register via the ``anemoi.inference.runners`` entry-point group:

.. code-block:: toml

    [project.entry-points."anemoi.inference.runners"]
    steady-state = "anemoi.plugins.ecmwf.inference.dynamics.steady_state:SteadyStateRunner"
"""

import logging

from anemoi.inference.runner import RunnerClasses
from anemoi.inference.runners.default import DefaultRunner
from anemoi.inference.tensors import TensorHandler

LOG = logging.getLogger(__name__)


class SteadyStateTensorHandler(TensorHandler):
    """TensorHandler that keeps dynamic forcings frozen at their initial values.

    On every autoregressive step the base-class implementation would reload
    dynamic forcings (insolation, cos/sin of local time, etc.) for the new
    date.  This subclass overrides that method to skip the reload entirely:
    the slots are simply marked as *already filled* in the bookkeeping array,
    leaving the tensor values from the initial step unchanged.
    """

    def add_dynamic_forcings_to_input_tensor(self, input_tensor_torch, state, dates, check):
        """Skip reloading dynamic forcings; mark their slots as already set."""
        for source in self.dynamic_forcings_providers:
            check[source.mask] = True
        return input_tensor_torch


class SteadyStateRunner(DefaultRunner):
    """Runner that uses the real model with dynamic forcings frozen at t=0.

    Dynamic forcings (e.g. insolation, local solar time trigonometric
    functions) are computed once from the initial date and then held constant
    throughout the entire rollout.  All other behaviour — model inference,
    pre/post/mid-processors, output — is identical to the default runner.

    Typical use-case: pair with a ``subtract_tendency`` mid-processor to
    isolate the anomaly response of an initial-condition perturbation.

    .. code-block:: yaml

        runner: steady-state

        mid_processors:
          - subtract_tendency:
              tend_pl_path: /path/to/tend_pl.grib
              tend_sfc_path: /path/to/tend_sfc.grib
              param_pl: ["z", "q", "t", "u", "v", "w"]
              level_pl: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
              param_sfc: ["msl", "10u", "10v", "2t"]
    """

    def __init__(self, config, *, classes=None) -> None:
        super().__init__(config, classes=RunnerClasses(tensor_handler=SteadyStateTensorHandler))
