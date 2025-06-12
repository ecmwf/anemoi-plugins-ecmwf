# anemoi-plugins-ecmwf

This repository contains all the plugins developed for anemoi by ECMWF.

**DISCLAIMER**
This project is **BETA** and will be **Experimental** for the foreseeable future.
Interfaces and functionality are likely to change, and the project itself may be scrapped.
**DO NOT** use this software in any project/software that is operational.

## Install

As this repo contains many repositories, see the underlying pages for information
about how to install.

## Conventions

When adding a plugin to this monorepo, add it under `plugins/{ANEMOI_PACKAGE}/{PLUGIN_NAME}`

i.e. for an inference source

```text
plugins
| --- inference
      | --- my_input
```

For best discoverability, the plugin should follow the naming convention,

```text
anemoi_{PACKAGE_NAME}_{SUB_INFORMATION}_{PLUGIN_NAME}
```

i.e. for the `my_input` above, if it is an inference input,

```text
anemoi_inference_input_my_input
```

## Versioning & Release

### pyproject.toml

To setup a new plugin for release and proper versioning, ensure you do the following,

In the `pyproject.toml` including the following to allow dynamic versioning, and replace the parts in `{}`.

```toml

[tool.setuptools_scm]

root = ".."
version_scheme = "post-release"
local_scheme = "no-local-version"
git_describe_command = "git describe --dirty --tags --long --match '{PACKAGE}-{NAME}-*[0-9]*'"
tag_regex = "^{PACKAGE}-{NAME}-(?P<version>[vV]?[0-9]+[^-]*)"
version_file = "src/{PACKAGE_PATH}/_version.py"
fallback_version = "0.0.0"
```

### release-please

Then to set it up with release-please, modify the `.release-please-config.yaml`. And your package,
and replace the parts in `{}`.

```yaml
packages: {
    "{PATH_TO_PACKAGE}": {
      "package-name": "{PACKAGED_NAME}",
      "component": "{PATH_EXCEPT_PLUGINS}"
    }
}
```

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
