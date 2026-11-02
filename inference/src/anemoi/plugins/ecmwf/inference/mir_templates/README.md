# An anemoi-inference mir grib templates plugin

Use `mir` to dynamically create grib templates of the correct geometry.

## Grids

The grid and area come from the template lookup, as for any named or regular grid.
A bare number such as `0.5` (what anemoi-inference makes of `[0.5, 0.5]`) is
read as `0.5/0.5`.

For unstructured output grids (e.g. stretched grids built with a cutout, or
Lambert grids, which are unstructured in GRIB) the provider builds the MIR
grid from the `latitudes` / `longitudes` of the output state. The template is
built from the lookup grid and area, and then checked against the state: if
the number of points differs from the number of output points (for a cutout
the lookup grid and area describe only one of the sources), or MIR cannot
build the grid or area, the template is rebuilt as an unstructured grid from
the state coordinates, dropping the area. If the lookup has no grid the state
coordinates are used directly.

If there is no grid and the state has no latitudes/longitudes, the provider
returns `None` so the next template provider is tried. Errors from MIR on the
unstructured grid are raised, as is a `ValueError` if the final template still
does not have one point per output point.

## Example

Use with the grib output

```yaml

output:
  grib:
    path: data.grib
    templates:
      - 'mir'

```
