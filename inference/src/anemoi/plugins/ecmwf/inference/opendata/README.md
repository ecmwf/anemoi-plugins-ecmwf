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
