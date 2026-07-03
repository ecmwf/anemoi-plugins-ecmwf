# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Iterator

import earthkit.data as ekd
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter
from anemoi.transform.filters.fields.matching import MatchingSpec


class VirtualTemperature(MatchingFieldsFilter):

    MATCHING = MatchingSpec(
        select="param",
        forward=("specific_humidity", "temperature"),
        skip_partial=True,
        return_inputs="all",
    )

    def __init__(
        self,
        *,
        virtual_temperature: str = "vtmp",
        specific_humidity: str = "q",
        temperature: str = "t",
    ):
        """Initialise the VirtualTemperature filter.

        Compute the virtual temperature from specific humidity and temperature,
        using
        ```
        vtmp = (1 + 0.61 * q) * t
        ```

        Parameters
        ----------
        virtual_temperature: str, optional
            Name of the virtual temperature parameter, by default "vtmp".
        specific_humidity : str, optional
            Name of the specific humidity parameter, by default "q".
        temperature : str, optional
            Name of the temperature parameter, by default "t".
        """
        self.specific_humidity = specific_humidity
        self.temperature = temperature
        self.virtual_temperature = virtual_temperature
        super().__init__()

    def forward_transform(self, specific_humidity: ekd.Field, temperature: ekd.Field) -> Iterator[ekd.Field]:
        vtmp = (1 + 0.61 * specific_humidity.to_numpy()) * temperature.to_numpy()
        yield self.new_field_from_numpy(vtmp, template=specific_humidity, param=self.virtual_temperature)
