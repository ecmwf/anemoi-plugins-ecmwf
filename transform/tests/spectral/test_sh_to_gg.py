# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import numpy as np
import pytest
from anemoi.plugins.ecmwf.transform.spectral.backends import BACKENDS
from anemoi.plugins.ecmwf.transform.spectral.utils import grid_to_pl
from anemoi.plugins.ecmwf.transform.spectral.utils import nspec_from_trunc
from anemoi.plugins.ecmwf.transform.spectral.utils import truncate_spectral

# backend_name fixture is provided by tests/spectral/conftest.py.


def _any_backend_available():
    return any(ok for ok, _ in (cls.available() for cls in BACKENDS.values()))


@pytest.mark.skipif(
    not _any_backend_available(),
    reason="No spectral backend available",
)
class TestShToGGIntegration:
    """Integration tests for spectral-to-gridpoint scalar transforms."""

    def _make_backend(self, grid, trunc, backend_name):
        from anemoi.plugins.ecmwf.transform.spectral.backends import make_backend

        return make_backend(grid, trunc, order=[backend_name])

    def test_zero_spectral_gives_zero_gridpoint(self, backend_name):
        """All-zero spectral coefficients produce zero gridpoint values."""
        trunc = 21
        grid = "O22"
        kloen = grid_to_pl(grid)
        ngptot = kloen.sum()
        backend = self._make_backend(grid, trunc, backend_name)

        nspec = nspec_from_trunc(trunc)
        spectral = np.zeros(nspec, dtype=np.float64)

        result = backend.sh_to_gg(spectral)

        assert result.shape[-1] == ngptot
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_constant_field(self, backend_name):
        """Setting only n=0,m=0 produces a uniform gridpoint field."""
        trunc = 21
        grid = "O22"
        backend = self._make_backend(grid, trunc, backend_name)

        nspec = nspec_from_trunc(trunc)
        spectral = np.zeros(nspec, dtype=np.float64)
        spectral[0] = 300.0  # n=0, m=0 real part (global mean)

        result = backend.sh_to_gg(spectral)

        assert np.isfinite(result).all()
        # All gridpoints should have the same value
        np.testing.assert_allclose(result, result.mean(), atol=1e-6)

    def test_nonzero_spectral_produces_nonzero_gridpoint(self, backend_name):
        """Random spectral coefficients produce non-zero gridpoint values."""
        trunc = 21
        grid = "O22"
        backend = self._make_backend(grid, trunc, backend_name)

        nspec = nspec_from_trunc(trunc)
        np.random.seed(42)
        spectral = np.random.randn(nspec) * 10.0

        result = backend.sh_to_gg(spectral)

        assert np.abs(result).max() > 0
        assert np.isfinite(result).all()

    def test_output_shape_matches_grid(self, backend_name):
        """Output number of gridpoints matches the target grid."""
        trunc = 21
        grid = "O22"
        kloen = grid_to_pl(grid)
        ngptot = kloen.sum()
        backend = self._make_backend(grid, trunc, backend_name)

        nspec = nspec_from_trunc(trunc)
        np.random.seed(42)
        spectral = np.random.randn(nspec) * 10.0

        result = backend.sh_to_gg(spectral)

        assert result.shape == (ngptot,)

    def test_batched_sh_to_gg(self, backend_name):
        """Batched transform handles multiple fields."""
        trunc = 21
        grid = "O22"
        kloen = grid_to_pl(grid)
        ngptot = kloen.sum()
        backend = self._make_backend(grid, trunc, backend_name)

        nspec = nspec_from_trunc(trunc)
        nfields = 3
        np.random.seed(123)
        spectral = np.random.randn(nfields, nspec) * 10.0

        result = backend.sh_to_gg(spectral)

        assert result.shape == (nfields, ngptot)
        assert np.isfinite(result).all()

    def test_batched_fields_are_independent(self, backend_name):
        """Each field in a batch is transformed independently."""
        trunc = 21
        grid = "O22"
        backend = self._make_backend(grid, trunc, backend_name)

        nspec = nspec_from_trunc(trunc)
        spectral_a = np.zeros(nspec, dtype=np.float64)
        spectral_b = np.zeros(nspec, dtype=np.float64)
        spectral_a[0] = 100.0
        spectral_b[0] = 200.0

        # Transform individually
        result_a = backend.sh_to_gg(spectral_a)
        result_b = backend.sh_to_gg(spectral_b)

        # Transform as a batch
        batched = np.vstack([spectral_a, spectral_b])
        result_batch = backend.sh_to_gg(batched)

        np.testing.assert_allclose(result_batch[0], result_a, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(result_batch[1], result_b, atol=1e-6, rtol=1e-6)

    def test_truncated_spectral_produces_valid_output(self, backend_name):
        """Pre-truncated spectral data still produces valid gridpoint output."""
        trunc_in = 42
        trunc_out = 21
        grid_out = "O22"

        nspec_in = nspec_from_trunc(trunc_in)
        np.random.seed(42)
        spectral = np.random.randn(nspec_in) * 10.0

        spectral_trunc = truncate_spectral(spectral, trunc_out)

        backend = self._make_backend(grid_out, trunc_out, backend_name)
        result = backend.sh_to_gg(spectral_trunc)

        kloen = grid_to_pl(grid_out)
        ngptot = kloen.sum()
        assert result.shape[-1] == ngptot
        assert np.isfinite(result).all()
