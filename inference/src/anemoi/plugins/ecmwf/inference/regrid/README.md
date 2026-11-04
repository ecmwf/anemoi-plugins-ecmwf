# An anemoi-inference regridding preprocessor

Use `mir` to regrid input fieldlists. Install with `pip install anemoi-plugins-ecmwf-inference[regrid]`.

Can be used with any `grid`/`area` spec for Global and LAM.

## Options

- `grid`: the target grid. One of
  - a grid string, e.g. `N320`, `O96`
  - a list of increments, e.g. `[0.25, 0.25]`
  - a named grid (see below)
  - `checkpoint` (see below)
  - a dict with `latitudes` and `longitudes`, given either as lists of values or as paths to `.npy` files
  - if omitted, the grid of the checkpoint metadata is used
- `area`: optional target area, as `north/west/south/east` or a list of the four values.

## Example

Use with the mars input

```yaml
input:
  mars:
    grid: O48

pre_processors:
  - regrid:
      grid: N320
      area: 89.785/0.0/-89.785/359.719
```

## Named Grids

Included within the `anemoi-plugins-ecmwf-transform` regrid module (`anemoi.plugins.ecmwf.transform.regrid.named`) are some extra named grids, that can be used in place of the `grid` key (names are case insensitive).

- meps: MetCoOp Ensemble Prediction System grid

## Checkpoint grid

Additionally, one can regrid directly to the coordinate arrays included in the checkpoint. This is best done via a
`pre_processor` in an `input`.

```yaml
input:
  mars:
    pre_processors:
      - regrid:
          grid: checkpoint
```

or if using a stretched grid model, or one with coords saved elsewhere in the supporting arrays, use `:` to break it up.
`checkpoint` uses the `latitudes` and `longitudes` supporting arrays, `checkpoint:source0` uses `source0/latitudes` and `source0/longitudes`.
An error is raised if the arrays are not in the checkpoint.

```yaml
input:
  mars:
    pre_processors:
      - regrid:
          grid: checkpoint:source0
```
