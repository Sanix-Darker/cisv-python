# cisv-python

Python bindings distribution for CISV (ctypes + nanobind).

## Upstream core

- cisv-core: https://github.com/Sanix-Darker/cisv-core

## Install nanobind package

```bash
cd bindings/python-nanobind
pip install .
```

## Benchmark Docker

```bash
docker build -t cisv-pynb-bench -f bindings/python-nanobind/benchmarks/Dockerfile .
docker run --rm --platform linux/amd64 --cpus=2 --memory=4g cisv-pynb-bench
```
