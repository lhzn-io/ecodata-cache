Tidal Harmonics (GOT4.10c / EOT20)
====================================

Overview
--------
Ecodata Cache introduces explicit tidal boundary condition support via
the `tides_tmd` fetcher. Harmonic components from **GOT4.10c** (NASA/GSFC)
or **EOT20** (DTU/DGFI) are fetched alongside HYCOM subtidal datasets.

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Model
     - Resolution
     - Provider
     - Notes
   * - GOT4.10c
     - ~0.5°
     - NASA GSFC
     - Default. 8 constituents.
   * - EOT20
     - ~0.125°
     - DTU Space / DGFI-TUM
     - Higher-resolution. 8 constituents.

Tidal Strategy
--------------
- **Subtidal state** (3D velocity, T, S, SSH): provided by HYCOM GOFS 3.1 (~9 km)
- **Tidal amplitudes & phases**: extracted by GOT4.10c/EOT20 on a 0.1° grid.

Usage
-----
POST to ``/api/v1/harmonics`` with the requested point coordinates and ``model_name`` field:

.. code-block:: json

   {
     "lons": [-74.0, -73.5],
     "lats": [40.5, 41.0],
     "model_name": "GOT4.10c"
   }

Set ``"model_name": "EOT20"`` to use the DTU/DGFI model instead.

Prerequisites
-------------
Model files must be downloaded before use. The fetcher raises a ``RuntimeError``
with download instructions if files are absent.

Set model directory via environment variable::

    export PYTMD_DATA_DIR=/path/to/tidal/models

Or place files in the default location: ``~/.pytmd/``

To download GOT4.10c::

    python -c "import pyTMD; pyTMD.io.model(/path/to/dir).from_database(GOT4.10c).download()"

Constituents
------------
Default set: M2, S2, N2, K2, K1, O1, P1, Q1 (8 major tidal constituents)

Caching
-------
Scalar fields are explicitly downloaded and returned directly.
