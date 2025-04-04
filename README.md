# rlaopt

<!-- [![formatter: docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter) -->
<!-- [![style: google](https://img.shields.io/badge/%20style-google-3666d6.svg)](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings) -->
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=udellgroup_rlaopt&metric=coverage&token=439522ce58af80988d93a4c197fc1f4da3f4e9b1)](https://sonarcloud.io/summary/new_code?id=udellgroup_rlaopt)

A package containing implementations of randomized linear algebra-based optimization algorithms for scientific computing and optimization.

## Installation

Please clone this repo. The package can be installed in a python environment in editable mode using the following command:

```bash
pip install -e .
```

We provide several environment variables to control the build process.

| Variable | Default | Description |
|----------|---------|-------------|
| `RLAOPT_CPU_ONLY` | `0` | Set to `1` to force CPU-only build |
| `RLAOPT_USE_CUDA` | `1` | Set to `0` to disable CUDA even if available |
| `RLAOPT_USE_OPENMP` | `1` | Set to `0` to disable OpenMP parallelization |
| `RLAOPT_DEBUG` | `0` | Set to `1` to build with debug symbols and no optimization |

Example usage:

```bash
RLAOPT_CPU_ONLY=1 pip install -e .
```

## Citation

If you find our work useful, please consider citing our paper:

```
TODO: add bibtex citation
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
