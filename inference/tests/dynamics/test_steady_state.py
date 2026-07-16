# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the steady-state runner plugin."""

from types import SimpleNamespace

import numpy as np
import torch
from anemoi.inference.tensors import TensorHandler
from anemoi.plugins.ecmwf.inference.dynamics.steady_state import SteadyStateRunner
from anemoi.plugins.ecmwf.inference.dynamics.steady_state import SteadyStateTensorHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(dynamic_masks):
    """Return a bare SteadyStateTensorHandler with mocked forcings providers."""
    handler = SteadyStateTensorHandler.__new__(SteadyStateTensorHandler)
    handler.dynamic_forcings_providers = [SimpleNamespace(mask=m) for m in dynamic_masks]
    return handler


# ---------------------------------------------------------------------------
# SteadyStateTensorHandler
# ---------------------------------------------------------------------------


class TestSteadyStateTensorHandler:
    def test_is_subclass_of_tensor_handler(self):
        assert issubclass(SteadyStateTensorHandler, TensorHandler)

    def test_check_marked_true_for_dynamic_forcings(self):
        """Dynamic-forcing slots must be marked as filled in the check array."""
        mask = np.array([2, 4])
        handler = _make_handler([mask])

        n_vars = 6
        tensor = torch.zeros(1, 1, 10, n_vars)
        check = np.zeros(n_vars, dtype=bool)

        handler.add_dynamic_forcings_to_input_tensor(tensor, state={}, dates=[], check=check)

        assert check[2] and check[4]
        assert not check[0] and not check[1] and not check[3] and not check[5]

    def test_tensor_values_unchanged(self):
        """The input tensor must NOT be modified — forcings are frozen."""
        mask = np.array([1, 3])
        handler = _make_handler([mask])

        original = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
        tensor = original.clone()
        check = np.zeros(6, dtype=bool)

        handler.add_dynamic_forcings_to_input_tensor(tensor, state={}, dates=[], check=check)

        assert torch.equal(tensor, original), "tensor was modified — forcings should be frozen"

    def test_multiple_providers_all_marked(self):
        """All providers' masks must be reflected in the check array."""
        handler = _make_handler([np.array([0]), np.array([3, 5])])

        check = np.zeros(6, dtype=bool)
        tensor = torch.zeros(1, 1, 4, 6)
        handler.add_dynamic_forcings_to_input_tensor(tensor, state={}, dates=[], check=check)

        assert check[0] and check[3] and check[5]
        assert not check[1] and not check[2] and not check[4]

    def test_no_providers_leaves_check_unchanged(self):
        """With no dynamic-forcings providers the check array stays all False."""
        handler = _make_handler([])

        check = np.zeros(4, dtype=bool)
        tensor = torch.zeros(1, 1, 4, 4)
        handler.add_dynamic_forcings_to_input_tensor(tensor, state={}, dates=[], check=check)

        assert not check.any()

    def test_returns_same_tensor_object(self):
        """The method must return the same tensor object (no copy)."""
        handler = _make_handler([np.array([0])])
        tensor = torch.zeros(1, 1, 4, 3)
        check = np.zeros(3, dtype=bool)
        assert handler.add_dynamic_forcings_to_input_tensor(tensor, state={}, dates=[], check=check) is tensor


# ---------------------------------------------------------------------------
# SteadyStateRunner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SteadyStateRunner
# ---------------------------------------------------------------------------


def test_steady_state_runner_subclasses_default_runner():
    """SteadyStateRunner must extend DefaultRunner."""
    from anemoi.inference.runners.default import DefaultRunner

    assert issubclass(SteadyStateRunner, DefaultRunner)


def test_steady_state_runner_uses_steady_state_tensor_handler(monkeypatch):
    """SteadyStateRunner.__init__ must pass SteadyStateTensorHandler via RunnerClasses."""
    from anemoi.inference.runner import RunnerClasses

    captured = {}
    original_init = SteadyStateRunner.__init__

    def fake_super_init(self, config, *, classes=None):
        captured["classes"] = classes

    monkeypatch.setattr(
        "anemoi.inference.runners.default.DefaultRunner.__init__",
        fake_super_init,
        raising=False,
    )

    runner = SteadyStateRunner.__new__(SteadyStateRunner)
    original_init(runner, config=object())

    assert isinstance(captured.get("classes"), RunnerClasses)
    assert captured["classes"].tensor_handler is SteadyStateTensorHandler


def test_steady_state_runner_is_registered():
    """steady-state must appear in the runner registry once the plugin is installed."""
    from anemoi.inference.runners import runner_registry

    assert runner_registry.is_registered("steady-state"), "'steady-state' not in runner_registry"
