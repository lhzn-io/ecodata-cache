import warnings
import logging
import pandas as pd
import numpy as np
import xarray as xr
import os
from typing import Optional
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

logger = logging.getLogger(__name__)


def get_metadata() -> dict:
    return {
        "id": "necofs",
        "name": "NECOFS FVCOM GOM7",
        "resolution_approx_m": 200.0,
        "type_desc": "Unstructured Triangular Mesh",
        "domain_bbox": [-77.0, 35.0, -65.0, 46.0],
    }


def supports_bbox(bbox: list[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    # Rough check for GOM3 bounds (NECOFS FVCOM)
    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        return False
    return True


NECOFS_GOM7_URL = "http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Forecasts/NECOFS_GOM7_FORECAST.nc"


def get_necofs_url(target_dt: pd.Timestamp) -> str:
    """
    Determine best NECOFS GOM7 URL by falling back to daily history archives.
    SMAST daily archives (e.g. 2026_03_03.nc) contain [Mar 2 01:00 to Mar 3 00:00].
    """
    import requests

    # NECOFS GOM7 daily history files contain data for the PREVIOUS day up to 00:00 of the CURRENT day.
    # Therefore to get data forward-looking from target_dt, we must ALWAYS fetch the file for the NEXT day.
    file_dt = target_dt.normalize() + pd.Timedelta(days=1)

    date_str = file_dt.strftime("%Y_%m_%d")
    history_url = f"http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Archive/necofs_history/NECOFS_GOM7_{date_str}.nc"

    try:
        resp = requests.get(history_url + ".dds", timeout=5)
        if resp.status_code == 200:
            logger.info(f"Using historic NECOFS GOM7 archive: {history_url}")
            return history_url
    except requests.RequestException:
        pass

    logger.info("Falling back to NECOFS GOM7 rolling forecast.")
    return NECOFS_GOM7_URL


def fetch_necofs_initial_conditions(
    target_date: str, bbox: list
) -> Optional[xr.Dataset]:
    """
    Fetches the 3D unstructured ocean state from NECOFS (GOM3),
    and regrids it to a structured Z-level grid suitable for Oceananigans.

    Args:
        target_date: ISO 8601 datestring (e.g. "2026-03-04T00:00:00Z")
        bbox: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Rough check for GOM3 bounds (NECOFS FVCOM)
    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        logger.info(
            f"Bounding box {bbox} is completely outside NECOFS GOM3 domain. Aborting fetch."
        )
        return None

    # Enforce UTC timezone naivety
    target_dt = pd.to_datetime(target_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    logger.info("Attempting to fetch initial conditions from NECOFS GOM3...")

    try:
        # Determine the correct URL based on target date (history archive vs rolling forecast)
        dap_url = get_necofs_url(target_dt)

        # Pydap might hang on SMAST, so we should rely on dispatcher catching/logging it,
        # or we could use requests to check if it's alive first.
        import requests

        try:
            # Quick alive check
            resp = requests.get(dap_url + ".dds", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"NECOFS server appears unreachable at {dap_url}: {e}")
            return None

        # Open the dataset
        # We must disable decode_times because GOM7 contains broken Itime/Itime2 variables
        # specifying "msec since 00:00:00" which cftime cannot parse.
        # We load without decoding, drop them, and decode the rest properly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_raw = xr.open_dataset(dap_url, engine="pydap", decode_times=False)
        drop_vars = [v for v in ["Itime", "Itime2"] if v in ds_raw.variables]
        ds = xr.decode_cf(ds_raw.drop_vars(drop_vars))

        # Extract unstructured coordinate variables
        lon = ds["lon"].values
        lat = ds["lat"].values
        lonc = ds["lonc"].values
        latc = ds["latc"].values

        _h = ds["h"].values  # noqa: F841 - needed for future depth mapping
        _siglay = ds["siglay"].values  # noqa: F841 - shape: (siglay, node)

        # Check nodes within bbox + buffer
        buffer = 0.05
        node_mask = (
            (lon >= min_lon - buffer)
            & (lon <= max_lon + buffer)
            & (lat >= min_lat - buffer)
            & (lat <= max_lat + buffer)
        )

        if not np.any(node_mask):
            logger.warning("No NECOFS nodes found within the padded bounding box.")
            return None

        # Temporal sub-selection
        try:
            ds_t = ds.sel(time=target_dt, method="nearest")
        except KeyError:
            logger.error(f"Cannot find time {target_dt} in NECOFS GOM3.")
            return None

        # Load required fields into memory (for the region or full domain depending on size)
        # Fetching full domain of one timestep might be faster/easier than fancy indexing with pydap
        logger.info("Downloading variables from NECOFS...")
        try:
            # We explicitly need node variables: temp, salinity, zeta
            # And element variables: u, v (and ww if available, but we'll stick to u, v)
            vars_to_get = ["temp", "salinity", "zeta", "u", "v"]
            ds_sub = ds_t[vars_to_get].compute()
            zeta = ds_sub["zeta"].values
            temp = ds_sub["temp"].values
            salt = ds_sub["salinity"].values
            u = ds_sub["u"].values
            v = ds_sub["v"].values
        except Exception as e:
            logger.error(f"Failed to extract variables from NECOFS: {e}")
            return None

        # Determine dimensions of regular grid
        # 0.002 degrees ~ roughly 200 meters resolution matching coastal NECOFS
        d_spacing = 0.002
        lon_rho = np.arange(min_lon, max_lon, d_spacing)
        lat_rho = np.arange(min_lat, max_lat, d_spacing)
        xi_rho = np.arange(len(lon_rho))
        eta_rho = np.arange(len(lat_rho))

        # Create target structured meshgrid
        lon_grid, lat_grid = np.meshgrid(lon_rho, lat_rho)

        logger.info("Building FVCOM interpolators for regridding...")

        # Node-based interpolator (temp, salt, zeta - defined at mesh nodes)
        pts_node = np.column_stack((lon, lat))
        tri_node = Delaunay(pts_node)

        # Element-based interpolator (u, v - defined at element centroids)
        pts_elem = np.column_stack((lonc, latc))
        tri_elem = Delaunay(pts_elem)

        target_pts = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

        logger.info("Interpolating variables to structured grid...")

        def interpolate_nodes(field_vals):
            interp = LinearNDInterpolator(tri_node, field_vals)
            return interp(target_pts).reshape(lon_grid.shape)

        def interpolate_elems(field_vals, strict_mask):
            interp = LinearNDInterpolator(tri_elem, field_vals)
            val = interp(target_pts).reshape(lon_grid.shape)
            val[strict_mask] = np.nan
            return val

        # 2D field
        zeta_interp = interpolate_nodes(zeta)
        land_mask = np.isnan(zeta_interp)

        s_rho_dim = ds_t.sizes.get("siglay", 45)
        s_rho = np.linspace(-1, 0, s_rho_dim)

        # Preallocate 3D arrays
        # shape: (s_rho, eta_rho, xi_rho)  - matching our target schema conventions
        ny, nx = len(lat_rho), len(lon_rho)
        nz = s_rho_dim

        temp_out = np.zeros((nz, ny, nx), dtype=np.float32)
        salt_out = np.zeros((nz, ny, nx), dtype=np.float32)
        u_out = np.zeros((nz, ny, nx), dtype=np.float32)
        v_out = np.zeros((nz, ny, nx), dtype=np.float32)

        for k in range(nz):
            # node based
            t_k = interpolate_nodes(temp[k, :])
            s_k = interpolate_nodes(salt[k, :])
            # elem based
            u_k = interpolate_elems(u[k, :], land_mask)
            v_k = interpolate_elems(v[k, :], land_mask)

            temp_out[k, :, :] = t_k
            salt_out[k, :, :] = s_k
            u_out[k, :, :] = u_k
            v_out[k, :, :] = v_k

        # Assemble Output Dataset matching general ROMS/DOPPIO output layout
        # (Variables: u, v, temp, salt, zeta, s_rho, lon_rho, lat_rho)

        ds_out = xr.Dataset(
            data_vars={
                "temp": (("s_rho", "eta_rho", "xi_rho"), temp_out),
                "salt": (("s_rho", "eta_rho", "xi_rho"), salt_out),
                "u": (("s_rho", "eta_rho", "xi_rho"), u_out),
                "v": (("s_rho", "eta_rho", "xi_rho"), v_out),
                "zeta": (("eta_rho", "xi_rho"), zeta_interp),
            },
            coords={
                "s_rho": s_rho,
                "eta_rho": eta_rho,
                "xi_rho": xi_rho,
                "lon_rho": (("eta_rho", "xi_rho"), lon_grid),
                "lat_rho": (("eta_rho", "xi_rho"), lat_grid),
            },
            attrs={"type": "NECOFS/FVCOM GOM3", "source": "UMass Dartmouth SMAST"},
        )

        logger.info(
            f"Successfully processed NECOFS data. Target grid dims: {dict(ds_out.sizes)}"
        )
        return ds_out

    except Exception as e:
        logger.error(f"Failed to fetch or process NECOFS data: {str(e)}")
        return None


def fetch_necofs_boundary_conditions(
    start_date: str, duration_hours: int, bbox: list[float]
) -> Optional[xr.Dataset]:
    """
    Fetches the 4D unstructured ocean state from NECOFS (GOM3/MASSBAY),
    and regrids it to a structured grid suitable for OBCs.
    Fetches multiple files if duration_hours spans multiple days.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        logger.info(f"Bounding box {bbox} outside NECOFS domain.")
        return None

    target_dt = pd.to_datetime(start_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    import requests

    d_spacing = 0.002
    lon_rho = np.arange(min_lon, max_lon, d_spacing)
    lat_rho = np.arange(min_lat, max_lat, d_spacing)
    lon_grid, lat_grid = np.meshgrid(lon_rho, lat_rho)
    target_pts = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    tri_node, tri_elem = None, None
    land_mask = None
    depths = None
    s_rho_dim = 45

    collected_times = []
    collected_u = []
    collected_v = []
    collected_temp = []
    collected_salt = []
    collected_zeta = []

    current_dt = target_dt
    hours_fetched = 0

    while hours_fetched < duration_hours:
        dap_url = get_necofs_url(current_dt)
        try:
            requests.get(dap_url + ".dds", timeout=10).raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"NECOFS server unreachable for {current_dt}: {e}")
            break

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds_raw = xr.open_dataset(dap_url, engine="pydap", decode_times=False)
            drop_vars = [v for v in ["Itime", "Itime2"] if v in ds_raw.variables]
            ds = xr.decode_cf(ds_raw.drop_vars(drop_vars))

            ds_times = pd.DatetimeIndex(ds.time.values)
            if ds_times.tz is not None:
                ds_times = ds_times.tz_convert("UTC").tz_localize(None)

            if tri_node is None:
                lon = ds["lon"].values
                lat = ds["lat"].values
                lonc = ds["lonc"].values
                latc = ds["latc"].values

                logger.info("Building NECOFS OBC interpolators...")
                pts_node = np.column_stack((lon, lat))
                tri_node = Delaunay(pts_node)
                pts_elem = np.column_stack((lonc, latc))
                tri_elem = Delaunay(pts_elem)

                s_rho_dim = ds.sizes.get("siglay", 45)
                depths = np.linspace(-50, 0, s_rho_dim)

                zeta_0 = ds["zeta"].isel(time=0).values
                zeta_grid = LinearNDInterpolator(tri_node, zeta_0)(target_pts).reshape(
                    lon_grid.shape
                )
                land_mask = np.isnan(zeta_grid)

            valid_mask = ds_times >= current_dt
            if not valid_mask.any():
                logger.warning(f"No valid times >= {current_dt} found in {dap_url}")
                current_dt += pd.Timedelta(days=1)
                continue

            start_idx = int(np.argmax(valid_mask))
            remaining_hours = duration_hours - hours_fetched
            file_remaining_steps = len(ds_times) - start_idx
            take_steps = min(remaining_hours, file_remaining_steps)

            end_idx = start_idx + take_steps - 1
            ds_t = ds.isel(time=slice(start_idx, end_idx + 1))
            nt = int(take_steps)
            nz = s_rho_dim
            ny = len(lat_rho)
            nx = len(lon_rho)

            logger.info(f"Extracting {nt} time steps from {dap_url} (parallelized)...")

            import concurrent.futures

            def process_time_step(t_idx, current_hours_fetched):
                logger.info(
                    f"Fetching time step {t_idx + 1}/{nt} for OBC (Grand Total: {current_hours_fetched + t_idx + 1}/{duration_hours})..."
                )
                # Note: PyDAP engine may occasionally hiccup with concurrent connections,
                # but usually per-timestep extraction on separate slices avoids direct overlap
                # if xarray isolates them, or we catch and retry if needed.
                u_t = ds_t["u"].isel(time=t_idx).values
                v_t = ds_t["v"].isel(time=t_idx).values
                temp_t = ds_t["temp"].isel(time=t_idx).values
                salt_t = ds_t["salinity"].isel(time=t_idx).values
                zeta_t = ds_t["zeta"].isel(time=t_idx).values

                interp_zeta = LinearNDInterpolator(tri_node, zeta_t)
                zeta_k = interp_zeta(target_pts).reshape(lon_grid.shape)
                zeta_k[land_mask] = np.nan

                u_out_t = np.zeros((nz, ny, nx), dtype=np.float32)
                v_out_t = np.zeros((nz, ny, nx), dtype=np.float32)
                temp_out_t = np.zeros((nz, ny, nx), dtype=np.float32)
                salt_out_t = np.zeros((nz, ny, nx), dtype=np.float32)

                for k in range(nz):
                    interp_u = LinearNDInterpolator(tri_elem, u_t[k, :])
                    interp_v = LinearNDInterpolator(tri_elem, v_t[k, :])
                    interp_temp = LinearNDInterpolator(tri_node, temp_t[k, :])
                    interp_salt = LinearNDInterpolator(tri_node, salt_t[k, :])

                    u_k = interp_u(target_pts).reshape(lon_grid.shape)
                    u_k[land_mask] = np.nan
                    v_k = interp_v(target_pts).reshape(lon_grid.shape)
                    v_k[land_mask] = np.nan

                    temp_k = interp_temp(target_pts).reshape(lon_grid.shape)
                    temp_k[land_mask] = np.nan
                    salt_k = interp_salt(target_pts).reshape(lon_grid.shape)
                    salt_k[land_mask] = np.nan

                    u_out_t[k, :, :] = u_k
                    v_out_t[k, :, :] = v_k
                    temp_out_t[k, :, :] = temp_k
                    salt_out_t[k, :, :] = salt_k

                return t_idx, u_out_t, v_out_t, temp_out_t, salt_out_t, zeta_k

            # Pre-allocate for ordered insertion
            u_results = [None] * nt
            v_results = [None] * nt
            temp_results = [None] * nt
            salt_results = [None] * nt
            zeta_results = [None] * nt

            max_workers = int(os.environ.get("ECODATA_CACHE_MAX_WORKERS", 4))
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(process_time_step, t, hours_fetched)
                    for t in range(nt)
                ]
                for future in concurrent.futures.as_completed(futures):
                    t_idx, u_out_t, v_out_t, temp_out_t, salt_out_t, zeta_k = (
                        future.result()
                    )
                    u_results[t_idx] = u_out_t
                    v_results[t_idx] = v_out_t
                    temp_results[t_idx] = temp_out_t
                    salt_results[t_idx] = salt_out_t
                    zeta_results[t_idx] = zeta_k

            collected_zeta.extend(zeta_results)
            collected_u.extend(u_results)
            collected_v.extend(v_results)
            collected_temp.extend(temp_results)
            collected_salt.extend(salt_results)

            hours_fetched += int(take_steps)
            collected_times.extend(ds_t.time.values)

            # Advance current_dt
            if end_idx + 1 < len(ds_times):
                current_dt = ds_times[end_idx + 1]
            else:
                current_dt = ds_times[-1] + pd.Timedelta(hours=1)

        except Exception as e:
            logger.error(f"Failed to process NECOFS OBC chunk: {e}")
            break

    if hours_fetched == 0:
        return None

    # Stack results
    u_out = np.stack(collected_u, axis=0)
    v_out = np.stack(collected_v, axis=0)
    temp_out = np.stack(collected_temp, axis=0)
    salt_out = np.stack(collected_salt, axis=0)
    zeta_out = np.stack(collected_zeta, axis=0)

    ds_out = xr.Dataset(
        data_vars={
            "u": (("time", "depth", "eta", "xi"), u_out),
            "v": (("time", "depth", "eta", "xi"), v_out),
            "temp": (("time", "depth", "eta", "xi"), temp_out),
            "salt": (("time", "depth", "eta", "xi"), salt_out),
            "zeta": (("time", "eta", "xi"), zeta_out),
        },
        coords={
            "time": collected_times,
            "depth": depths,
            "eta": lat_rho,
            "xi": lon_rho,
        },
        attrs={"type": "NECOFS/FVCOM GOM3 OBC", "source": "UMass Dartmouth SMAST"},
    )
    return ds_out
