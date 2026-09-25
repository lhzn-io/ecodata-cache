# <img src="static/logo.svg" height="36" valign="middle" alt="Logo" /> EcodataCache

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<div align="center">
  <img src="docs/source/_static/ecodata-cache-inventory-screenshot.png" alt="EcodataCache Dashboard" />
</div>

`EcodataCache` is a highly-optimized Earth-system proxy, designed for coupling multi-domain environmental physics into high-fidelity unified simulations.

It operates primarily as a high-speed data acquisition and spatiotemporal transformation service, sitting between large-scale operational forecasting endpoints (NOAA, Copernicus ERA5, HRRR, HYCOM, IOOS Regional Nodes) and native solvers like `Oceananigans.jl`.

From serving tidal harmonic constituents to delivering atmospheric boundary layer fields, this application strips away the complexity of federated, heterogeneous climate data.

## Goals

The primary goal of Ecodata Cache is to provide clean, standardized ocean and atmospheric forcing boundaries for high-fidelity coastal hydrodynamic models. We achieve this by:

- Operating as an invisible, self-managing proxy service that orchestrates downloading and caching of remote data.
- Stripping away the quirks of individual forecasting nodes by harmonizing temporal alignment, grid structures, and variable naming conventions.
- Generating deterministic cache hashes to ensure simulation runs are 100% physically reproducible over time.

## Features

- **Decoupled System Architecture**: The microservice guarantees reproducible datasets by deterministic hashing, providing standardized local representations of external APIs.
- **Continuous Coupled Hindcasts**: Support for 45-to-60-day continuous hindcasts by dynamically stitching historical and operational HYCOM experiments.
- **Volumetric Forcing API**: ERA5 pressure-level fetching for u, v, w, T, Z, and specific humidity.
- **Tidal Extrapolation**: Access to high-resolution global harmonic predictors via PyTMD (`GOT4.10c` & `EOT20`) returned via standard API endpoints without complex spatial assembly.
- **Temporal Alignment**: Automated CF-compliant hourly interpolation across heterogeneous upstream timestamps and grid frequencies.

## Current Coverage

- **Oceanic Forcing**: NYOFS, NECOFS, DBOFS, HYCOM (Operational & Historical).
- **Atmospheric Forcing**: HRRR (3km surface), ERA5 (30km volumetric & surface retro-scaling).
- **Tidal Harmonics**: GOT4.10c (Default, ~0.5°), EOT20 (~0.125°).

## Roadmap

- **Enhanced IOOS Coverage**: Expanding our data fetchers to support more regional nodes across the West Coast (WCOFS) and Gulf of Mexico (NGOFS2).
- **Improved Caching Policies**: Implementing dynamic cache invalidation based on NOAA operational forecast updates to ensure realtime predictions stay synchronized.
- **Variable Expansions**: Providing native spatiotemporal transformations for wave spectra and biogeochemical tracers.

## Architecture overview

The service exposes HTTP APIs (via FastAPI/Uvicorn) which:

1. Deterministically hash bounding boxes and time windows to generate an automated local Zarr cache (`~/.cache/ecodata-cache`).
2. Dispatch fetch operations via tiered fallback chains for high-availability. (e.g. HRRR -> ERA5T -> ERA5 for atmospheric conditions).
3. Standardize dimensions, bounds, floating-point precision, and timezone offsets, providing a direct ingestion path for downstream coupled engines.

## Setup

```bash
uv sync
pre-commit install

# Development (single worker, run from the repository root):
uv run uvicorn service.ecodata_serve.main:app --port 9598

# Service launcher (supports --host/--port/--workers/--reload):
uv run python service/run_server.py --host 0.0.0.0 --port 9598 --workers 4
```

## Docker Environment

The `Dockerfile` defaults to `ubuntu:24.04`, which is multi-arch. This service is
CPU-only, so the default works unchanged on x86_64 and on arm64 edge hardware;
`BASE_IMAGE` exists for cases that need a different base.

```bash
# x86_64 and arm64 alike (default)
docker build -t ecodata-cache .

docker run --rm -p 9598:9598 ecodata-cache
```

On NVIDIA Jetson under JetPack 7 (L4T R39, Ubuntu 24.04), keep the default base.
An earlier revision of this file recommended
`--build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-base:r36.2.0`; that was wrong on two
counts. The `l4t-*` image family was never published beyond `r36.4.0`
(JetPack 6.1), so there is no R39 equivalent to move to, and this service needs
no CUDA in the image regardless. Where a CUDA base genuinely is required, the
JetPack 7 path is the unified `nvcr.io/nvidia/cuda:<ver>-*-ubuntu24.04` images,
which carry real arm64 manifests, combined with CDI device injection
(`--device nvidia.com/gpu=all`). Note that `--gpus all` is rejected on Tegra.

## Licensing

This project is licensed under the Apache 2.0 License.
