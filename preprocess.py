import sys
import time

import numpy as np
import xarray as xr

from config import (EXPERIMENTS, MODELS, MONTH_SLICE, N_YEARS, TARGET_MODEL,
                    WORK_DIR, raw_files)


def load_native(model, exp):
    files = raw_files(model, exp)
    if not files:
        raise FileNotFoundError(f"{model} {exp}: no files found")

    slabs, lat, lon, total = [], None, None, 0
    need = MONTH_SLICE.stop
    for f in files:
        if total >= need:
            break
        with xr.open_dataset(f, decode_times=False) as ds:
            take = min(ds.sizes["time"], need - total)
            slabs.append(ds["o3"].isel(lev=0, time=slice(0, take)).values)
            lat, lon = ds.lat.values, ds.lon.values
            total += take

    data = np.concatenate(slabs, axis=0)[MONTH_SLICE]
    if data.shape[0] != N_YEARS * 12:
        raise ValueError(f"{model} {exp}: expected {N_YEARS * 12} months, got {data.shape[0]}")
    return data, lat, lon


def target_grid():
    files = raw_files(TARGET_MODEL, "piClim-control")
    if not files:
        raise FileNotFoundError(f"{TARGET_MODEL} piClim-control: no files found")
    with xr.open_dataset(files[0], decode_times=False) as ds:
        return ds.lat.values, ds.lon.values


def regrid(data, lat, lon, tlat, tlon):
    if (len(lat) == len(tlat) and len(lon) == len(tlon)
            and np.allclose(lat, tlat, atol=1e-3) and np.allclose(lon, tlon, atol=1e-3)):
        return data

    lon_ext = np.concatenate([[lon[-1] - 360.0], lon, [lon[0] + 360.0]])
    data_ext = np.concatenate([data[:, :, -1:], data, data[:, :, :1]], axis=2)

    da = xr.DataArray(
        data_ext,
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(data.shape[0]), "lat": lat, "lon": lon_ext},
    )
    return da.interp(lat=tlat, lon=tlon, method="linear").values


def main():
    force = "--force" in sys.argv
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tlat, tlon = target_grid()
    np.savez(WORK_DIR / "grid.npz", lats=tlat, lons=tlon)
    print(f"common grid {len(tlat)}x{len(tlon)}, simulation years 2-30 (n={N_YEARS})")

    for model in MODELS:
        for exp in EXPERIMENTS:
            out = WORK_DIR / f"surf_{model}_{exp}_monthly.npy"
            if out.exists() and not force:
                print(f"[skip] {out.name}")
                continue
            t0 = time.time()
            data, lat, lon = load_native(model, exp)
            data = regrid(data, lat, lon, tlat, tlon)
            arr = data.reshape(N_YEARS, 12, len(tlat), len(tlon)).astype(np.float32)
            np.save(out, arr)
            print(f"[done] {out.name} {arr.shape} "
                  f"global mean {np.nanmean(arr) * 1e9:.2f} ppb ({time.time() - t0:.0f}s)")

    for exp in EXPERIMENTS:
        out = WORK_DIR / f"mme_{exp}_monthly.npy"
        if out.exists() and not force:
            print(f"[skip] {out.name}")
            continue
        stack = np.stack([np.load(WORK_DIR / f"surf_{m}_{exp}_monthly.npy") for m in MODELS])
        np.save(out, np.nanmean(stack, axis=0).astype(np.float32))
        print(f"[done] {out.name}")


if __name__ == "__main__":
    main()
