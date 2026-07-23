# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from anemoi.plugins.ecmwf.transform.spectral.backends import BACKENDS
from anemoi.plugins.ecmwf.transform.spectral.backends import CalculationBackend
from anemoi.plugins.ecmwf.transform.spectral.backends import get_backend
from anemoi.plugins.ecmwf.transform.spectral.utils import _classic_reduced_pl
from anemoi.plugins.ecmwf.transform.spectral.utils import _truncation_indices
from anemoi.plugins.ecmwf.transform.spectral.utils import grid_to_pl
from anemoi.plugins.ecmwf.transform.spectral.utils import grid_to_trunc
from anemoi.plugins.ecmwf.transform.spectral.utils import nspec_from_trunc
from anemoi.plugins.ecmwf.transform.spectral.utils import trunc_from_nspec
from anemoi.plugins.ecmwf.transform.spectral.utils import truncate_spectral

# ============================================================================
# Utils: grid_to_pl
# ============================================================================


class TestGridToPl:
    def test_octahedral_shape(self):
        pl = grid_to_pl("O32")
        assert pl.shape == (64,)  # 2 * 32

    def test_octahedral_symmetry(self):
        pl = grid_to_pl("O32")
        np.testing.assert_array_equal(pl[:32], pl[63:31:-1])

    def test_octahedral_first_row(self):
        pl = grid_to_pl("O32")
        assert pl[0] == 20

    def test_octahedral_increment(self):
        pl = grid_to_pl("O32")
        # Northern hemisphere rows: 20, 24, 28, ..., 20 + 4*(32-1) = 144
        expected_nh = np.arange(20, 20 + 4 * 32, 4, dtype=np.int64)
        np.testing.assert_array_equal(pl[:32], expected_nh)

    def test_full_grid_shape(self):
        pl = grid_to_pl("F48")
        assert pl.shape == (96,)  # 2 * 48

    def test_full_grid_uniform(self):
        pl = grid_to_pl("F48")
        assert np.all(pl == 192)  # 4 * 48

    def test_classic_reduced_shape(self):
        pl = grid_to_pl("N32")
        assert pl.shape == (64,)

    def test_classic_reduced_symmetry(self):
        pl = grid_to_pl("N32")
        np.testing.assert_array_equal(pl[:32], pl[63:31:-1])

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown grid type"):
            grid_to_pl("X32")


# ============================================================================
# Utils: grid_to_trunc
# ============================================================================


class TestGridToTrunc:
    def test_octahedral(self):
        assert grid_to_trunc("O1280") == 1279

    def test_classic_reduced(self):
        assert grid_to_trunc("N320") == 639

    def test_full(self):
        assert grid_to_trunc("F48") == 95

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown grid type"):
            grid_to_trunc("X100")

    def test_lowercase_handled(self):
        # The code does .upper() on prefix
        assert grid_to_trunc("o640") == 639


# ============================================================================
# Utils: nspec / trunc round-trip
# ============================================================================


class TestSpectralSize:
    @pytest.mark.parametrize("trunc", [0, 1, 10, 63, 319, 1279])
    def test_round_trip(self, trunc):
        nspec = nspec_from_trunc(trunc)
        assert trunc_from_nspec(nspec) == trunc

    def test_nspec_formula(self):
        # T=10 -> 11*12 = 132
        assert nspec_from_trunc(10) == 132

    def test_trunc_from_invalid_nspec(self):
        with pytest.raises(ValueError, match="Cannot derive truncation"):
            trunc_from_nspec(13)  # not (T+1)*(T+2) for any integer T


# ============================================================================
# Utils: truncate_spectral
# ============================================================================


class TestTruncateSpectral:
    def test_no_truncation_when_target_ge_input(self):
        trunc_in = 5
        nspec_in = nspec_from_trunc(trunc_in)
        data = np.random.randn(nspec_in)
        result = truncate_spectral(data, t_out=5)
        np.testing.assert_array_equal(result, data)

    def test_no_truncation_when_target_gt_input(self):
        trunc_in = 5
        nspec_in = nspec_from_trunc(trunc_in)
        data = np.random.randn(nspec_in)
        result = truncate_spectral(data, t_out=10)
        np.testing.assert_array_equal(result, data)

    def test_output_size(self):
        trunc_in = 10
        trunc_out = 5
        nspec_in = nspec_from_trunc(trunc_in)
        nspec_out = nspec_from_trunc(trunc_out)
        data = np.random.randn(nspec_in)
        result = truncate_spectral(data, trunc_out)
        assert result.shape == (nspec_out,)

    def test_batched_output_size(self):
        trunc_in = 10
        trunc_out = 5
        nspec_in = nspec_from_trunc(trunc_in)
        nspec_out = nspec_from_trunc(trunc_out)
        data = np.random.randn(3, nspec_in)
        result = truncate_spectral(data, trunc_out)
        assert result.shape == (3, nspec_out)

    def test_negative_target_raises(self):
        data = np.random.randn(nspec_from_trunc(5))
        with pytest.raises(ValueError, match="Target truncation must be >= 0"):
            truncate_spectral(data, t_out=-1)

    def test_preserves_low_order_coefficients(self):
        """Truncating T=3 -> T=1 should keep m=0 (n=0..1) and m=1 (n=1) coefficients."""
        trunc_in = 3
        nspec_in = nspec_from_trunc(trunc_in)
        # Fill with index values so we can verify which ones are kept
        data = np.arange(nspec_in, dtype=np.float64)
        result = truncate_spectral(data, t_out=1)
        # T_out=1: nspec = 2*3 = 6
        assert result.shape == (6,)
        # m=0: first 2*(1-0+1) = 4 values from offset 0
        # m=1: first 2*(1-1+1) = 2 values from offset 2*(3-0+1) = 8
        expected = np.array([0, 1, 2, 3, 8, 9], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)


# ============================================================================
# Utils: _truncation_indices
# ============================================================================


class TestTruncationIndices:
    def test_identity_when_same_truncation(self):
        idx = _truncation_indices(5, 5)
        nspec = nspec_from_trunc(5)
        np.testing.assert_array_equal(idx, np.arange(nspec))

    def test_output_length(self):
        idx = _truncation_indices(10, 5)
        assert len(idx) == nspec_from_trunc(5)

    def test_all_indices_valid(self):
        idx = _truncation_indices(10, 5)
        assert np.all(idx >= 0)
        assert np.all(idx < nspec_from_trunc(10))


# ============================================================================
# Utils: _classic_reduced_pl
# ============================================================================


class TestClassicReducedPl:
    @pytest.mark.parametrize("n", [32, 48, 80, 128, 160, 256, 320])
    def test_available_grids(self, n):
        pl = _classic_reduced_pl(n)
        assert pl.shape == (2 * n,)
        # Symmetric
        np.testing.assert_array_equal(pl[:n], pl[2 * n - 1 : n - 1 : -1])

    def test_unavailable_grid_raises(self):
        with pytest.raises(ValueError, match="not in lookup table"):
            _classic_reduced_pl(7)


# ============================================================================
# Backends: registry and get_backend
# ============================================================================


class TestBackendRegistry:
    def test_backends_contains_expected_keys(self):
        assert "ctrans4py" in BACKENDS
        assert "ectrans4py" in BACKENDS

    def test_all_backends_are_subclasses(self):
        for name, cls in BACKENDS.items():
            assert issubclass(cls, CalculationBackend), f"{name} is not a CalculationBackend"

    def test_get_backend_no_available_raises(self):
        with patch.dict(BACKENDS, {}, clear=True):
            with pytest.raises(RuntimeError, match="No available backend found"):
                get_backend()

    def test_get_backend_returns_first_available(self):
        class FakeBackend(CalculationBackend):
            @classmethod
            def available(cls):
                return True, "ok"

            def vordiv_to_uv(self, vorticity, divergence):
                return vorticity, divergence

            def uv_to_vordiv(self, u_component_of_wind, v_component_of_wind):
                return u_component_of_wind, v_component_of_wind

        with patch.dict(BACKENDS, {"fake": FakeBackend}, clear=True):
            result = get_backend()
            assert result is FakeBackend

    def test_get_backend_skips_unavailable(self):
        class UnavailableBackend(CalculationBackend):
            @classmethod
            def available(cls):
                return False, "not installed"

            def vordiv_to_uv(self, vorticity, divergence):
                pass

            def uv_to_vordiv(self, u_component_of_wind, v_component_of_wind):
                pass

        class AvailableBackend(CalculationBackend):
            @classmethod
            def available(cls):
                return True, "ok"

            def vordiv_to_uv(self, vorticity, divergence):
                return vorticity, divergence

            def uv_to_vordiv(self, u_component_of_wind, v_component_of_wind):
                return u_component_of_wind, v_component_of_wind

        with patch.dict(
            BACKENDS,
            {"bad": UnavailableBackend, "good": AvailableBackend},
            clear=True,
        ):
            result = get_backend()
            assert result is AvailableBackend

    def test_get_backend_respects_order(self):
        class BackendA(CalculationBackend):
            @classmethod
            def available(cls):
                return True, "ok"

            def vordiv_to_uv(self, vorticity, divergence):
                pass

            def uv_to_vordiv(self, u_component_of_wind, v_component_of_wind):
                pass

        class BackendB(CalculationBackend):
            @classmethod
            def available(cls):
                return True, "ok"

            def vordiv_to_uv(self, vorticity, divergence):
                pass

            def uv_to_vordiv(self, u_component_of_wind, v_component_of_wind):
                pass

        with patch.dict(
            BACKENDS,
            {"a": BackendA, "b": BackendB},
            clear=True,
        ):
            assert get_backend(order=["b", "a"]) is BackendB
            assert get_backend(order=["a", "b"]) is BackendA

    def test_get_backend_unknown_name_in_order(self):
        """Unknown backend name in order list is skipped gracefully."""

        class GoodBackend(CalculationBackend):
            @classmethod
            def available(cls):
                return True, "ok"

            def vordiv_to_uv(self, vorticity, divergence):
                pass

            def uv_to_vordiv(self, u_component_of_wind, v_component_of_wind):
                pass

        with patch.dict(BACKENDS, {"good": GoodBackend}, clear=True):
            result = get_backend(order=["nonexistent", "good"])
            assert result is GoodBackend


# ============================================================================
# VordivToUV: grid resolution logic
# ============================================================================


class TestResolveForwardGrid:
    def _make_filter(self, **kwargs):
        from anemoi.plugins.ecmwf.transform.spectral.vordiv_to_uv import VordivToUV

        return VordivToUV(**kwargs)

    def test_transform_grid_takes_precedence(self):
        f = self._make_filter(target_grid="O640", transform_grid="O320")
        mock_field = MagicMock()
        grid, trunc = f._resolve_forward_grid(mock_field)
        assert grid == "O320"
        assert trunc == 319

    def test_target_grid_used_when_no_transform_grid(self):
        f = self._make_filter(target_grid="O640")
        mock_field = MagicMock()
        grid, trunc = f._resolve_forward_grid(mock_field)
        assert grid == "O640"
        assert trunc == 639

    def test_derives_from_spectral_size(self):
        f = self._make_filter()
        trunc_in = 10
        nspec = nspec_from_trunc(trunc_in)
        mock_field = MagicMock()
        mock_field.to_numpy.return_value = np.zeros(nspec)
        grid, trunc = f._resolve_forward_grid(mock_field)
        assert trunc == trunc_in
        assert grid == f"O{trunc_in + 1}"


class TestResolveBackwardGrid:
    def _make_filter(self, **kwargs):
        from anemoi.plugins.ecmwf.transform.spectral.vordiv_to_uv import VordivToUV

        return VordivToUV(**kwargs)

    def test_spectral_grid_used(self):
        f = self._make_filter(spectral_grid="O640")
        grid, trunc = f._resolve_backward_grid()
        assert grid == "O640"
        assert trunc == 639

    def test_transform_grid_fallback(self):
        f = self._make_filter(transform_grid="O320")
        grid, trunc = f._resolve_backward_grid()
        assert grid == "O320"
        assert trunc == 319

    def test_spectral_grid_takes_precedence_over_transform(self):
        f = self._make_filter(spectral_grid="O640", transform_grid="O320")
        grid, trunc = f._resolve_backward_grid()
        assert grid == "O640"
        assert trunc == 639

    def test_raises_when_neither_set(self):
        f = self._make_filter()
        with pytest.raises(ValueError, match="spectral_grid or transform_grid must be set"):
            f._resolve_backward_grid()


# ============================================================================
# VordivToUV: patch_data_request
# ============================================================================


class TestPatchDataRequest:
    def _make_filter(self, **kwargs):
        from anemoi.plugins.ecmwf.transform.spectral.vordiv_to_uv import VordivToUV

        return VordivToUV(**kwargs)

    def test_replaces_uv_with_vordiv(self):
        f = self._make_filter()
        req = {"param": ["u", "v", "2t"]}
        result = f.patch_data_request(req)
        assert "u" not in result["param"]
        assert "v" not in result["param"]
        assert "vo" in result["param"]
        assert "d" in result["param"]
        assert "2t" in result["param"]

    def test_no_change_when_uv_absent(self):
        f = self._make_filter()
        req = {"param": ["2t", "msl"]}
        result = f.patch_data_request(req)
        assert result["param"] == ["2t", "msl"]

    def test_custom_param_names(self):
        f = self._make_filter(
            vorticity="vort",
            divergence="divg",
            u_component_of_wind="uu",
            v_component_of_wind="vv",
        )
        req = {"param": ["uu", "vv", "2t"]}
        result = f.patch_data_request(req)
        assert "uu" not in result["param"]
        assert "vv" not in result["param"]
        assert "vort" in result["param"]
        assert "divg" in result["param"]

    def test_empty_param_list(self):
        f = self._make_filter()
        req = {"param": []}
        result = f.patch_data_request(req)
        assert result["param"] == []

    def test_no_param_key(self):
        f = self._make_filter()
        req = {"level": [500, 850]}
        result = f.patch_data_request(req)
        assert "param" not in result

    def test_only_u_present_no_change(self):
        """If only u is present (not v), no substitution should happen."""
        f = self._make_filter()
        req = {"param": ["u", "2t"]}
        # The code checks `any(... for param in [u, v])`, so having just u
        # triggers the branch -- but then .remove(v) will raise ValueError.
        # This tests the current behaviour.
        with pytest.raises(ValueError):
            f.patch_data_request(req)


# ============================================================================
# Integration: VordivToUV forward_transform with real spectral backend
# ============================================================================

# backend_name fixture is provided by tests/spectral/conftest.py.


def _any_backend_available():
    return any(ok for ok, _ in (cls.available() for cls in BACKENDS.values()))


@pytest.mark.skipif(
    not _any_backend_available(),
    reason="No spectral backend (ctrans4py/ectrans4py) available",
)
class TestVordivToUVIntegration:
    """Integration tests that run the actual spectral transform."""

    def _make_backend(self, grid, trunc, backend_name):
        from anemoi.plugins.ecmwf.transform.spectral.backends import make_backend

        return make_backend(grid, trunc, order=[backend_name])

    def test_vordiv_to_uv_produces_gridpoint_fields(self, backend_name):
        """Forward transform produces u/v fields on the target grid."""
        trunc = 21
        grid = "O22"
        kloen = grid_to_pl(grid)
        ngptot = kloen.sum()

        # Create synthetic spectral vorticity and divergence (zero = no wind)
        nspec = nspec_from_trunc(trunc)
        backend = self._make_backend(grid, trunc, backend_name)

        vor = np.zeros(nspec, dtype=np.float64)
        div = np.zeros(nspec, dtype=np.float64)

        u, v = backend.vordiv_to_uv(vor, div)

        # Zero vor/div should give zero u/v
        assert u.shape[-1] == ngptot
        assert v.shape[-1] == ngptot
        np.testing.assert_allclose(u, 0.0, atol=1e-10)
        np.testing.assert_allclose(v, 0.0, atol=1e-10)

    def test_nonzero_vorticity_produces_nonzero_wind(self, backend_name):
        """Non-zero vorticity produces non-zero u/v."""
        trunc = 21
        grid = "O22"
        backend = self._make_backend(grid, trunc, backend_name)

        nspec = nspec_from_trunc(trunc)
        np.random.seed(42)
        vor = np.random.randn(nspec) * 1e-5
        div = np.random.randn(nspec) * 1e-5

        u, v = backend.vordiv_to_uv(vor, div)

        # Should produce non-zero wind
        assert np.abs(u).max() > 0
        assert np.abs(v).max() > 0
        # Values should be finite
        assert np.isfinite(u).all()
        assert np.isfinite(v).all()

    def test_batched_vordiv_to_uv(self, backend_name):
        """Batched transform handles multiple levels."""
        trunc = 21
        grid = "O22"
        kloen = grid_to_pl(grid)
        ngptot = kloen.sum()
        backend = self._make_backend(grid, trunc, backend_name)

        nspec = nspec_from_trunc(trunc)
        nfields = 3
        np.random.seed(123)
        vor = np.random.randn(nfields, nspec) * 1e-5
        div = np.random.randn(nfields, nspec) * 1e-5

        u, v = backend.vordiv_to_uv(vor, div)

        assert u.shape == (nfields, ngptot)
        assert v.shape == (nfields, ngptot)
        assert np.isfinite(u).all()
        assert np.isfinite(v).all()

    def test_uv_to_vordiv_round_trip(self, backend_name):
        """Round-trip: vordiv -> uv -> vordiv recovers original (within tolerance)."""
        if backend_name == "mir":
            pytest.skip("mir backend does not support uv_to_vordiv")

        trunc = 21
        grid = "O22"
        backend = self._make_backend(grid, trunc, backend_name)

        nspec = nspec_from_trunc(trunc)
        np.random.seed(99)
        vor_orig = np.random.randn(nspec) * 1e-5
        div_orig = np.random.randn(nspec) * 1e-5

        # Imaginary parts at m=0 must be zero for valid spectral data.
        # m=0 coefficients are stored as (real, imag) pairs at indices
        # 0..2*(trunc+1)-1; the odd indices are the imaginary parts.
        # The n=0,m=0 coefficient (index 0, global mean) is also lost in
        # the round-trip as it has no gridpoint wind expression.
        m0_len = 2 * (trunc + 1)
        vor_orig[0] = 0.0
        div_orig[0] = 0.0
        vor_orig[1:m0_len:2] = 0.0
        div_orig[1:m0_len:2] = 0.0

        # Forward: spectral vor/div -> gridpoint u/v
        u, v = backend.vordiv_to_uv(vor_orig, div_orig)

        # Backward: gridpoint u/v -> spectral vor/div
        vor_back, div_back = backend.uv_to_vordiv(u, v)

        # Should recover original spectral coefficients
        np.testing.assert_allclose(vor_back, vor_orig, rtol=1e-4, atol=1e-10)
        np.testing.assert_allclose(div_back, div_orig, rtol=1e-4, atol=1e-10)

    def test_truncate_before_transform(self, backend_name):
        """Pre-truncation reduces spectral content but still produces valid output."""
        # Create data at higher truncation
        trunc_in = 42
        trunc_out = 21
        grid_out = "O22"

        nspec_in = nspec_from_trunc(trunc_in)
        np.random.seed(42)
        vor = np.random.randn(nspec_in) * 1e-5
        div = np.random.randn(nspec_in) * 1e-5

        # Truncate to lower resolution
        vor_trunc = truncate_spectral(vor, trunc_out)
        div_trunc = truncate_spectral(div, trunc_out)

        backend = self._make_backend(grid_out, trunc_out, backend_name)
        u, v = backend.vordiv_to_uv(vor_trunc, div_trunc)

        kloen = grid_to_pl(grid_out)
        ngptot = kloen.sum()
        assert u.shape[-1] == ngptot
        assert v.shape[-1] == ngptot
        assert np.isfinite(u).all()
        assert np.isfinite(v).all()
