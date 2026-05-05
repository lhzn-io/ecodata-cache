# <img src="static/logo.svg" height="36" valign="middle" alt="Logo" /> EcodataCache

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
uv run uvicorn service.ecodata_serve.main:app --port 9598
```

## Docker Environment

The `Dockerfile` defaults to `ubuntu:24.04` (x86_64).
Override `BASE_IMAGE` for other hardware (like edge compute):

```bash
# x86_64 (default)
docker build -t ecodata-cache .

# NVIDIA Jetson AGX Orin (JetPack 6.x)
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-base:r36.2.0 \
  -t ecodata-cache .

docker run --rm -p 9598:9598 ecodata-cache
```

## Licensing

This project is licensed under the Apache 2.0 License.
