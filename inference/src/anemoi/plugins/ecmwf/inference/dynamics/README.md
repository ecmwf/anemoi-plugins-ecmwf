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

### Notes

1. The `array_overlay` can take a `npy` file in addition to the example shown above.
    It will be regridded from an N320 grid, so use latxlon
2. The `modify_value` can also take a `npy` file, but is expected to be of the right node shape.
