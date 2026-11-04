# An anemoi-inference polytope input plugin

Initialise `anemoi-inference` with polytope.

## Install

```python

pip install anemoi-plugins-ecmwf-inference[polytope]

```

## Usage

This input can be used like any other for anemoi inference, below is shown an example config.

> [!TIP]
> The following example requires, `anemoi-models==0.3.1` and `flash_attn` installed.

```yaml

checkpoint:
  huggingface: ecmwf/aifs-single-1.0

date: 2020-01-01

input: polytope
```

To run, just like any other

```bash
anemoi-inference run config.yaml
```

## Options

The input extends the `mars` input, so its options (`variables`, `patches`, `pre_processors`, `purpose`) and any extra keys added to the request are accepted. Additionally:

- `collection`: the polytope collection to retrieve from. Defaults to `ecmwf-mars`. It is the main argument, so `input: {polytope: ecmwf-mars}` is equivalent to the form below.

```yaml
input:
  polytope:
    collection: ecmwf-mars
```

Unless set in the config, requests use `expver=0001`, `class=od`, `stream=oper` and `step=0`, with `grid` and `area` taken from the checkpoint. Configured `patches` are applied last.
