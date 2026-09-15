# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Anemoi Inference Output Multio Plugin"""

import logging
import os

from .multio_output import MultioOutputPlugin as MultioOutputPlugin

LOG = logging.getLogger(__name__)

# Importing pygribjump will call dlopen on libgribjump.so.
# When this happens, the gribjump plugin registers a callback function with the FDB library.
# This function is responsible for generating the .gribjump index files in FDB, and is called whenever FDB archives a field.
# Traditionally, gribjump is dynamically loaded by eckit::Main's plugin mechanism,
# but this makes certain assumptions about the relative installation paths of the libraries which are true for the C++ bundles,
# but not for python wheels installed into site_packages.

if os.getenv("FDB_ENABLE_GRIBJUMP", "0") == "1":
    try:
        import pygribjump  # type: ignore

        LOG.info("gribjump version: %s", pygribjump.__version__)
    except ImportError:
        LOG.error("gribjump is not installed. FDB writes will not be indexed.")
