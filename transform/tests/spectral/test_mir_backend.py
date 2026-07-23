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

# Check if MIR is available
pytest.importorskip("mir", reason="MIR not available")
pytest.importorskip("eccodes", reason="eccodes not available")


class TestMIRBackend:
    """Test MIR backend for spectral transforms."""

    def test_mir_backend_registered(self):
        """MIR backend is in the registry."""
        assert "mir" in BACKENDS

    def test_mir_backend_available(self):
        """MIR backend reports as available."""
        mir_backend = BACKENDS["mir"]
        available, msg = mir_backend.available()
        assert available, f"MIR should be available: {msg}"

    def test_mir_backend_instantiation(self):
        """MIR backend can be instantiated."""
        mir_backend = BACKENDS["mir"]
        trunc = 21
        kloen = grid_to_pl("O22")
        backend = mir_backend(kloen, trunc, grid="O22")
        assert backend.trunc == trunc
        assert backend.grid == "O22"

    def test_mir_sh_to_gg_single_field(self):
        """MIR backend transforms single spectral field to gridpoint."""
        mir_backend = BACKENDS["mir"]
        trunc = 21
        kloen = grid_to_pl("O22")
        backend = mir_backend(kloen, trunc, grid="O22")

        # Create a simple spectral field (constant + small perturbation)
        nspec = nspec_from_trunc(trunc)
        spectral = np.zeros(nspec, dtype=np.float64)
        spectral[0] = 300.0  # Mean value
        spectral[1:5] = 0.1  # Small perturbation

        # Transform to gridpoint
        gridpoint = backend.sh_to_gg(spectral)

        # Check output shape
        assert gridpoint.ndim == 1
        assert len(gridpoint) == kloen.sum()

        # Check that values are reasonable
        assert np.isfinite(gridpoint).all()
        # Mean should be preserved (roughly)
        assert abs(gridpoint.mean() - 300.0) < 50.0

    def test_mir_sh_to_gg_multiple_fields(self):
        """MIR backend transforms multiple spectral fields to gridpoint."""
        mir_backend = BACKENDS["mir"]
        trunc = 21
        kloen = grid_to_pl("O22")
        backend = mir_backend(kloen, trunc, grid="O22")

        # Create multiple spectral fields
        nspec = nspec_from_trunc(trunc)
        nfields = 3
        spectral = np.zeros((nfields, nspec), dtype=np.float64)
        for i in range(nfields):
            spectral[i, 0] = 250.0 + i * 10.0  # Different mean for each field

        # Transform to gridpoint
        gridpoint = backend.sh_to_gg(spectral)

        # Check output shape
        assert gridpoint.shape == (nfields, kloen.sum())

        # Check that values are reasonable
        assert np.isfinite(gridpoint).all()
        for i in range(nfields):
            assert abs(gridpoint[i].mean() - (250.0 + i * 10.0)) < 50.0

    def test_mir_vordiv_to_uv_produces_valid_output(self):
        """MIR backend vordiv_to_uv produces finite gridpoint u/v."""
        mir_backend = BACKENDS["mir"]
        trunc = 21
        kloen = grid_to_pl("O22")
        backend = mir_backend(kloen, trunc, grid="O22")

        nspec = nspec_from_trunc(trunc)
        np.random.seed(42)
        vor = np.random.randn(nspec) * 1e-5
        div = np.random.randn(nspec) * 1e-5

        u, v = backend.vordiv_to_uv(vor, div)

        ngptot = kloen.sum()
        assert u.shape[-1] == ngptot
        assert v.shape[-1] == ngptot
        assert np.isfinite(u).all()
        assert np.isfinite(v).all()
        assert np.abs(u).max() > 0
        assert np.abs(v).max() > 0

    def test_mir_uv_to_vordiv_not_implemented(self):
        """MIR backend raises NotImplementedError for uv_to_vordiv."""
        mir_backend = BACKENDS["mir"]
        trunc = 21
        kloen = grid_to_pl("O22")
        backend = mir_backend(kloen, trunc, grid="O22")

        ngptot = kloen.sum()
        u = np.zeros(ngptot)
        v = np.zeros(ngptot)

        with pytest.raises(NotImplementedError, match="does not support uv_to_vordiv"):
            backend.uv_to_vordiv(u, v)
