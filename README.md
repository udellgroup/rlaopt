# rlaopt

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=udellgroup_rlaopt&metric=coverage&token=439522ce58af80988d93a4c197fc1f4da3f4e9b1)](https://sonarcloud.io/summary/new_code?id=udellgroup_rlaopt)

A package containing implementations of randomized linear algebra-based optimization algorithms for scientific computing and optimization.

> [!WARNING]
> This package is under active development. The API may change frequently, and the code may not be stable. Use at your own risk.

## Installation

Install the latest release from PyPI with pip:

```bash
pip install rlaopt
```

Or add it to a uv-managed project:

```bash
uv add rlaopt
```

## Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone this repository, and create the locked development environment with:

```bash
uv sync
```

Run commands inside the locked environment with `uv run`, for example:

```bash
uv run pytest
```

## Releasing

The version in `pyproject.toml` is the single source of truth. Maintainers prepare a short-lived release branch with `uv version X.Y.Z --no-sync`, merge it into `main`, and publish a GitHub Release tagged `vX.Y.Z`. Publishing the GitHub Release validates, attests, and uploads the package to PyPI through Trusted Publishing.

See [RELEASING.md](https://github.com/udellgroup/rlaopt/blob/main/RELEASING.md) for the complete setup, packaging, versioning, release-note, and recovery procedures.

## Citation

If you find our work useful, please consider citing our paper:

```
TODO: add bibtex citation
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/udellgroup/rlaopt/blob/main/LICENSE) file for details.
