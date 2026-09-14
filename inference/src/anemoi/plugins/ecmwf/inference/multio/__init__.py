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

if os.getenv("FDB_ENABLE_GRIBJUMP", "0") == "1":
    # As gribjump through a wheel install cannot be auto discovered by fdb, it must be imported here
    try:
        import pygribjump  # type: ignore

        LOG.info("gribjump version: %s", pygribjump.__version__)
    except ImportError:
        LOG.error("gribjump is not installed. FDB writes will not be indexed.")
