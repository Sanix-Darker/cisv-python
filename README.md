# cisv-python

![License](https://img.shields.io/badge/license-MIT-blue)

Python bindings distribution for CISV (ctypes + nanobind).

## Features

- Python CSV parsing with C/SIMD backend
- `parse_file`, `parse_string`, `count_rows`
- Iterator API for large files
- Parallel/fast modes in nanobind variant

## Installation

```bash
git clone https://github.com/Sanix-Darker/cisv-python
cd cisv-python
make -C core all
cd bindings/python-nanobind
pip install .
```

## Python API

### Basic example

```python
import cisv
rows = cisv.parse_file('data.csv', delimiter=',', trim=True)
```

### Detailed example (iterator + early exit)

```python
import cisv
for row in cisv.open_iterator('large.csv'):
    if row[0] == 'stop':
        break
```

More runnable examples: [`examples/`](./examples)

## Benchmarks

```bash
docker build -t cisv-pynb-bench -f bindings/python-nanobind/benchmarks/Dockerfile .
docker run --rm --platform linux/amd64 --cpus=2 --memory=4g cisv-pynb-bench
```

## Upstream Core

- cisv-core: https://github.com/Sanix-Darker/cisv-core
