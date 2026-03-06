# cisv-python
[![CI](https://github.com/Sanix-Darker/cisv-python/actions/workflows/ci.yml/badge.svg)](https://github.com/Sanix-Darker/cisv-python/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/cisv.svg)](https://pypi.org/project/cisv/)
[PyPI Package](https://pypi.org/project/cisv/)

![License](https://img.shields.io/badge/license-MIT-blue)

Python binding distribution for CISV using the nanobind implementation (`bindings/python-nanobind`) with SIMD-accelerated core parsing.

## Features

- Native C-backed parser exposed to Python
- `parse_file`, `parse_string`, and `count_rows` APIs
- Iterator API (`CisvIterator` / `open_iterator`) for memory-efficient streaming
- Parallel parse mode for large files
- Early-exit iteration support for pipeline-style processing

## Supported Python Versions

- Minimum supported version: Python 3.10
- Tested in CI on Python 3.10 and 3.12

## Installation

### From PyPI

```bash
pip install cisv
```

### From source

```bash
git clone --recurse-submodules https://github.com/Sanix-Darker/cisv-python
cd cisv-python
make -C core all
cd bindings/python-nanobind
pip install .
```

## Core Dependency (Submodule)

This repository tracks `cisv-core` via the `./core` git submodule.

To fetch the latest `cisv-core` (main branch) in your local clone:

```bash
git submodule update --init --remote --recursive
```

CI and release workflows also run this update command, so new `cisv-core` releases are pulled automatically during builds.

## Quick Start

```python
import cisv

rows = cisv.parse_file("data.csv", delimiter=",", trim=True)
print(rows[0])
```

## API Examples

### Parse a file with options

```python
import cisv

rows = cisv.parse_file(
    "data.csv",
    delimiter=",",
    quote='"',
    trim=True,
    skip_empty_lines=True,
)
```

### Parse in parallel

```python
import cisv

rows = cisv.parse_file("large.csv", parallel=True)
```

### Row counting without full parse

```python
import cisv

count = cisv.count_rows("large.csv")
print(count)
```

### Iterator mode for huge files

```python
import cisv

with cisv.CisvIterator("very_large.csv", trim=True) as it:
    for row in it:
        if row and row[0] == "STOP":
            break
        # process row
```

### Convenience iterator helper

```python
import cisv

for row in cisv.open_iterator("data.csv", delimiter=",", trim=True):
    print(row)
```

## Examples Directory

Runnable examples are available in [`examples/`](./examples):

- `basic.py`
- `iterator.py`
- `sample.csv`

## Benchmarks

```bash
docker build -t cisv-pynb-bench -f bindings/python-nanobind/benchmarks/Dockerfile .
docker run --rm --platform linux/amd64 --cpus=2 --memory=4g cisv-pynb-bench
```

The benchmark output includes both full parse and iterator paths (including `cisv-iterator`).

## Upstream Core

- cisv-core: https://github.com/Sanix-Darker/cisv-core
