# anemoi-plugins-ecmwf

This repository contains all the plugins developed for anemoi by ECMWF.

**DISCLAIMER**
This project is **BETA** and will be **Experimental** for the foreseeable future.
Interfaces and functionality are likely to change, and the project itself may be scrapped.
**DO NOT** use this software in any project/software that is operational.

## Install

As this repo contains many repositories, see the underlying pages for information
about how to install.

| Package | Directory | Install |
|---------|-----------|---------|
| `anemoi-plugins-ecmwf-inference` | [`inference/`](inference/README.md) | `pip install anemoi-plugins-ecmwf-inference[PLUGIN]` |
| `anemoi-plugins-ecmwf-transform` | [`transform/`](transform/README.md) | `pip install anemoi-plugins-ecmwf-transform[PLUGIN]` |

## Conventions

When adding a plugin to this monorepo, add it under `{ANEMOI_PACKAGE}/src/.../{PLUGIN_NAME}`

i.e. for an inference source

```text
inference
| --- src/.../
      | --- my_input
```

For best discoverability, new plugins should follow this naming convention:

- the directory name is the plugin name in snake_case, for example `my_input`;
- the entry point and the `optional-dependencies` extra use the same plugin name, in lower case and without the package prefix, for example `my_input`.

This is guidance for new plugins; some existing plugins predate it and keep their current names.

## Versioning & Release

### pyproject.toml

To setup a new plugin for release and proper versioning, ensure you do the following,

In the `pyproject.toml` for the plugin package include the new plugin in the `optional-dependencies`,
and in the `entry-points`

## License

```text
Copyright 2024-2025, ECMWF.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation
nor does it submit to any jurisdiction.
```
