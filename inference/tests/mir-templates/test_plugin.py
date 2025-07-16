# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.inference.grib.templates import create_template_provider
from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider


def test_registration():
    template_provider = create_template_provider(None, "mir")  # type: ignore
    assert isinstance(
        template_provider, MirTemplatesProvider
    ), "Template provider should be an instance of MirTemplatesProvider"


if __name__ == "__main__":
    test_registration()
