# An anemoi-inference opendata input plugin

Initialise `anemoi-inference` with ECMWF Opendata.

## Install

```python

pip install anemoi-plugins-ecmwf-inference[opendata]

```

## Usage

This input can be used like any other for anemoi inference, below is shown an example config.

> [!TIP]
> The following example requires, `anemoi-models==0.3.1` and `flash_attn` installed.

```yaml

checkpoint:
  huggingface: ecmwf/aifs-single-1.0

date: 2020-01-01

input: opendata
```

To run, just like any other

```bash
anemoi-inference run config.yaml
```

## Options

The input extends the `mars` input, so its options (`variables`, `patches`, `pre_processors`, `purpose`) and any extra keys added to the request are accepted. Additionally:

- `source`: where to download from, one of `ecmwf` (default), `aws`, `google` or `azure`.
- `grid` and `area`: default to those of the checkpoint.

```yaml
input:
  opendata:
    source: aws
```

### Notes

1. Requests are always made with `type=fc`.
2. Configured `patches` are applied to each request after the plugin's own adjustments, as in the `mars` and `polytope` inputs.
3. `use_grib_paramid` is not supported and is ignored with a warning.
4. Geopotential on pressure levels (`z`) is requested as `gh` and converted to `z`.
5. Soil temperature and moisture (`stl1`-`stl3`, `swvl1`-`swvl3`) are requested as `sot` and `vsw` at the matching level, and renamed back afterwards.
6. Fields are regridded to the checkpoint grid with MIR.
