# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import io
import logging

import earthkit.data as ekd
import numpy as np

from .base import CalculationBackend

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MIR backend (experimental)
# ---------------------------------------------------------------------------


class mir(CalculationBackend):
    """Backend using MIR (Meteorological Interpolation and Regridding) library.

    EXPERIMENTAL: This backend uses MIR for spectral transforms by encoding
    numpy arrays to GRIB and using MIR's spectral transform capabilities.
    Uses GribMemoryInput for efficient in-memory processing.
    Primarily for benchmarking MIR vs ectrans performance.

    Limitations:
    - ``vordiv_to_uv`` is not implemented (MIR's vod2uv requires special GRIB structure).
    - ``uv_to_vordiv`` is not implemented (no direct transform in MIR).
    - Requires creating temporary GRIB messages for each transform.
    - May have batching inefficiencies with multiple fields.
    """

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            import eccodes  # type: ignore[reportMissingImports]  # noqa: F401
            import mir  # type: ignore[reportMissingImports]  # noqa: F401

            return True, "mir and eccodes are available"
        except Exception as e:
            return False, f"mir backend not available: {e}"

    def _spectral_to_grib(self, spectral_data: np.ndarray, param_id: int = 130) -> bytes:
        """Encode spectral coefficients to a GRIB2 message.

        Parameters
        ----------
        spectral_data : np.ndarray
            Spectral coefficients, shape (nspec2,).
        param_id : int
            GRIB parameter ID (default 130 = temperature).

        Returns
        -------
        bytes
            GRIB2 message.
        """
        import eccodes as ec

        sample = ec.codes_grib_new_from_samples("sh_pl_grib2")
        try:
            ec.codes_set(sample, "pentagonalResolutionParameterJ", self.trunc)
            ec.codes_set(sample, "pentagonalResolutionParameterK", self.trunc)
            ec.codes_set(sample, "pentagonalResolutionParameterM", self.trunc)
            ec.codes_set(sample, "paramId", param_id)
            ec.codes_set(sample, "packingType", "spectral_complex")
            ec.codes_set(sample, "JS", self.trunc)
            ec.codes_set(sample, "KS", self.trunc)
            ec.codes_set(sample, "MS", self.trunc)
            ec.codes_set(sample, "bitsPerValue", 24)
            ec.codes_set_values(sample, spectral_data.astype(np.float64))
            return ec.codes_get_message(sample)
        finally:
            ec.codes_release(sample)

    def _transform_via_mir(self, grib_bytes: bytes, vod2uv: bool = False) -> io.BytesIO:
        """Transform GRIB spectral field(s) to gridpoint using MIR.

        Uses BytesIO buffers directly with job.execute() which correctly
        handles multiple GRIB messages in a single pass.

        Parameters
        ----------
        grib_bytes : bytes
            Concatenated GRIB2 message(s).
        vod2uv : bool
            If True, add vod2uv=1 to the MIR job (input must be vo/d pairs).

        Returns
        -------
        io.BytesIO
            Output buffer containing regridded GRIB messages.
        """
        import mir

        input_buffer = io.BytesIO(grib_bytes)
        output_buffer = io.BytesIO()
        job = mir.Job()
        job.set("grid", self.grid)
        if vod2uv:
            job.set("vod2uv", "1")
        job.execute(input_buffer, output_buffer)
        output_buffer.seek(0)
        return output_buffer

    def vordiv_to_uv(self, vorticity: np.ndarray, divergence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Spectral vorticity/divergence to grid-point u/v via MIR.

        Encodes vo/d as interleaved GRIB pairs and uses MIR's vod2uv option
        via MultiDimensionalGribFileInput(path, 2) which reads pairs together.
        Output is u/v pairs in level order.
        """
        import os
        import tempfile

        import mir as mir_lib

        squeezed = vorticity.ndim == 1
        vor = np.atleast_2d(np.ascontiguousarray(vorticity, dtype=np.float64))
        div = np.atleast_2d(np.ascontiguousarray(divergence, dtype=np.float64))

        # Encode interleaved vo/d pairs to a temp file
        # MIR vod2uv requires MultiDimensionalGribFileInput(path, 2) to read
        # vo/d messages together as a single multi-dimensional field.
        tmpf = tempfile.NamedTemporaryFile(suffix=".grib", delete=False)
        try:
            for i in range(vor.shape[0]):
                tmpf.write(self._spectral_to_grib(vor[i], param_id=138))  # vo
                tmpf.write(self._spectral_to_grib(div[i], param_id=155))  # d
            tmpf.close()

            inp = mir_lib.MultiDimensionalGribFileInput(tmpf.name, 2)
            output_buffer = io.BytesIO()
            job = mir_lib.Job()
            job.set("grid", self.grid)
            job.set("vod2uv", "1")
            job.execute(inp, output_buffer)
            output_buffer.seek(0)

            # Decode output u/v fields
            result = ekd.from_source("memory", output_buffer.getvalue())

            # Result is u/v pairs: u[0], v[0], u[1], v[1], ...
            nfields = vor.shape[0]
            u_list = []
            v_list = []
            for i in range(nfields):
                u_list.append(result[2 * i].to_numpy(flatten=True, dtype=np.float32))
                v_list.append(result[2 * i + 1].to_numpy(flatten=True, dtype=np.float32))

            u = np.vstack(u_list)
            v = np.vstack(v_list)

            if squeezed:
                return u.squeeze(0), v.squeeze(0)
            return u, v
        finally:
            os.unlink(tmpf.name)

    def uv_to_vordiv(
        self, u_component_of_wind: np.ndarray, v_component_of_wind: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grid-point u/v to spectral vorticity/divergence.

        NOTE: This operation is not implemented for the MIR backend.

        Raises
        ------
        NotImplementedError
            Always, as this operation is not supported.
        """
        raise NotImplementedError("mir backend does not support uv_to_vordiv (backward transform)")

    def sh_to_gg(self, scalars: np.ndarray) -> np.ndarray:
        """Spectral scalar field(s) to grid-point via MIR.

        Encodes spectral arrays to GRIB, passes all messages to MIR in one
        job.execute() call via BytesIO, then decodes the gridpoint output.

        Parameters
        ----------
        scalars : np.ndarray
            Spectral coefficients, shape ``(nfields, nspec2)`` or ``(nspec2,)``.

        Returns
        -------
        np.ndarray
            Grid-point values, shape ``(nfields, ngptot)`` or ``(ngptot,)``.
        """
        squeezed = scalars.ndim == 1
        q = np.atleast_2d(np.ascontiguousarray(scalars, dtype=np.float64))

        # Encode all fields to GRIB and concatenate
        messages = []
        for i in range(q.shape[0]):
            messages.append(self._spectral_to_grib(q[i]))
        grib_bytes = b"".join(messages)

        # Transform all fields in one MIR pass
        output_buffer = self._transform_via_mir(grib_bytes)

        # Decode output gridpoint fields
        output_bytes = output_buffer.getvalue()
        result = ekd.from_source("memory", output_bytes)

        gridpoint = np.vstack([f.to_numpy(flatten=True, dtype=np.float32) for f in result])

        if squeezed:
            return gridpoint.squeeze(0)
        return gridpoint
