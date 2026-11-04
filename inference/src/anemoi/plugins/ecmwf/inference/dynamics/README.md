# An anemoi-inference initial condition preprocessor

Modify initial conditions to explore model dynamics

## Install

```python

pip install anemoi-plugins-ecmwf-inference[dynamics]

```

## Usage

This preprocessor can be used like any other for anemoi inference, below is shown an example config.

> [!TIP]
> The following example requires, `anemoi-models==0.3.1` and `flash_attn` installed.

```yaml

checkpoint:
  huggingface: ecmwf/aifs-single-1.0

date: 2020-01-01

input: mars

pre_processors:
  - array_overlay:
      overlay: "https://get.ecmwf.int/repository/anemoi/assets/duck.jpg"
      fields:
          - {"shortName": "2t"}
      rescale: 10
      method: "add"
      invert: true
  - modify_value:
      value: 200
      fields:
          - {"shortName": "msl"}
      method: "subtract"
```

To run, just like any other

```bash
anemoi-inference run config.yaml
```

### Options

`array_overlay` (pre-processor):

- `overlay`: local path or URL of an image, or a local path to a `.npy` file (`.npy` URLs are not supported).
- `fields`: list of metadata filters (for example `{"shortName": "2t"}`); a field is modified if it matches any of them.
- `rescale`: scale applied to the overlay, default `1`.
- `method`: one of `add` (default), `multiply` or `replace`.
- `invert`: if `true`, white is treated as transparent. Default `false`.

`modify_value` (pre-processor):

- `fields`: list of metadata filters, as above.
- `value`: a number, or the path of a `.npy` file.
- `method`: one of `add` (default), `subtract`, `multiply`, `divide` or `replace`.

### Other plugins

- `regression-perturbation` (pre-processor): applies a localised climatological regression perturbation (Hakim and Masanam, 2023) to the initial conditions. Required options are `season` (`DJF` or `JAS`), `data_path`, `data_grid`, `ylat`, `xlon`, `xlev`, `locrad`, `amp`, `param_pl`, `level_pl` and `param_sfc`. Optional ones are `method` (as for `modify_value`, default `add`), `file_glob_suffix` (default `.grib`), `max_samples`, `rescale` (default `1.0`) and `grid`. The perturbation is computed on `data_grid`. If `grid` is unset (the default) no regridding is done, so the input fields must already be on `data_grid`. If `grid` is set to a grid specification (e.g. `O96`), the perturbation is regridded from `data_grid` to it with `earthkit-regrid`; `grid: checkpoint` regrids to the grid recorded in the checkpoint metadata.
- `subtract_tendency` (mid-processor): subtracts precomputed tendencies from the model output at each step, using `tend_pl_path`, `tend_sfc_path`, `param_pl`, `level_pl` and `param_sfc`.
- `steady-state` (runner): the default runner with dynamic forcings (for example insolation) frozen at their initial values. It is intended to be paired with `subtract_tendency`.

```yaml
runner: steady-state

mid_processors:
  - subtract_tendency:
      tend_pl_path: /path/to/tend_pl.grib
      tend_sfc_path: /path/to/tend_sfc.grib
      param_pl: ["z", "q", "t", "u", "v", "w"]
      level_pl: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
      param_sfc: ["msl", "10u", "10v", "2t"]
```

### Notes

1. The `array_overlay` image is padded to the 1440x721 aspect ratio, resized to 1440x721 and regridded from a regular 0.25 degree latitude-longitude grid to the field grid. A `.npy` overlay is processed in the same way, so it should be an image-like array of shape (height, width, channels). Pillow is required.
2. The `modify_value` can also take a `npy` file, but is expected to be of the right node shape.
