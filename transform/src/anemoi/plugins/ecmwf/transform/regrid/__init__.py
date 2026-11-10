# (C) Copyright 2025-2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

# eccodes needs eckit-geo to resolve unstructured grids (uuidOfHGrid), eg. MIR output on named grids
os.environ.setdefault("ECCODES_ECKIT_GEO", "1")
# eckit-geo only records unstructured grids by uid in its cache (grid.yaml) with caching on,
# without it another process cannot decode the output of MIR on such grids.
# Must be set before eckit-geo is first used in the process.
os.environ.setdefault("ECKIT_GEO_CACHING", "1")

from .regrid import MIRRegrid

__all__ = ["MIRRegrid"]
