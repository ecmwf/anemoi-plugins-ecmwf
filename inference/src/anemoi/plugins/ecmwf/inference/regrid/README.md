# An anemoi-inference regridding preprocessor

Use `mir` / `earthkit-regrid` to regrid input fieldlists.

Can be used with any `grid`/`area` spec for Global and LAM.

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

Included within the regrid module are some extra named grids, that can be used in place of the `grid` key.

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

```yaml
input:
  mars:
    pre_processors:
      - regrid:
          grid: checkpoint:source0
```
