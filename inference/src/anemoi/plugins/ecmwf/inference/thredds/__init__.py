# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .profiles import available_profiles
from .profiles import get_profile
from .profiles.schema import ThreddsEntry
from .profiles.schema import ThreddsProfile
from .thredds_profiles import ThreddsProfilesInput

__all__ = [
    "ThreddsProfile",
    "ThreddsEntry",
    "ThreddsProfilesInput",
    "get_profile",
    "available_profiles",
]
