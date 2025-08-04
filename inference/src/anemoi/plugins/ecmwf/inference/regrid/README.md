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
