# An anemoi-inference mir grib templates plugin

Use `mir` to dynamically create grib templates of the correct geometry.

## Grids

The grid (and area) come from the template lookup, as for any named or regular grid.

For unstructured output grids (e.g. stretched grids built with a cutout, or
Lambert grids, which are unstructured in GRIB) the provider builds the MIR
grid from the `latitudes` / `longitudes` of the output state. This is done when

- the lookup has no grid, or
- the lookup grid has no area and either cannot be built by MIR or has a
  different number of points than the output state.

Otherwise the lookup grid is used unchanged. If there is no grid and the state
has no latitudes/longitudes, the provider returns `None` so the next template
provider is tried.

## Example

Use with the grib output

```yaml

output:
  grib:
    path: data.grib
    templates:
      - 'mir'

```
