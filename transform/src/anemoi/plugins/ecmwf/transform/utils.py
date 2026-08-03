# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import earthkit.data as ekd


def fields_to_numpy_parallel(fields: "ekd.FieldList", dtype=np.float32, max_workers=None) -> np.ndarray:
    """Decode multiple GRIB fields to numpy array in parallel.

    GRIB decoding via eccodes is thread-safe on different handles and releases
    the GIL, enabling near-linear speedup with ThreadPoolExecutor.  Using
    dtype=float32 (default) triggers eccodes' native codes_get_float_array for
    grid_ccsds and grid_simple packing, halving decode time and memory.

    Parameters
    ----------
    fields : earthkit.data.FieldList
        Input fields to decode.
    dtype : numpy.dtype, optional
        Data type for output array, by default np.float32.  Use np.float64 if
        downstream operations require double precision (e.g. ectrans).
    max_workers : int, optional
        Maximum number of worker threads. If None, uses min(8, os.cpu_count()).
        Can be overridden by setting ANEMOI_GRIB_DECODE_THREADS environment variable.

    Returns
    -------
    numpy.ndarray
        Array of shape (len(fields), npoints) with decoded field values.
        Returns empty array of shape (0, 0) if fields is empty.
    """
    nfields = len(fields)
    if nfields == 0:
        return np.empty((0, 0), dtype=dtype)

    # Single field: no threading overhead
    if nfields == 1:
        data = fields[0].to_numpy(flatten=True, dtype=dtype)
        return data.reshape(1, -1)

    # Check environment variable for max_workers override
    if max_workers is None:
        env_workers = os.environ.get("ANEMOI_GRIB_DECODE_THREADS")
        if env_workers is not None:
            try:
                max_workers = int(env_workers)
            except ValueError:
                pass

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)

    # Decode first field to determine size
    first_data = fields[0].to_numpy(flatten=True, dtype=dtype)
    npoints = len(first_data)

    # Preallocate result array
    result = np.empty((nfields, npoints), dtype=dtype)
    result[0, :] = first_data

    # Decode remaining fields in parallel
    def decode_field(i):
        return i, fields[i].to_numpy(flatten=True, dtype=dtype)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, data in executor.map(lambda idx: decode_field(idx), range(1, nfields)):
            result[i, :] = data

    return result
