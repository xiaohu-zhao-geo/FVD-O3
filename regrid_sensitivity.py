import numpy as np
import pandas as pd

from config import (EXPERIMENTS, MODELS, N_YEARS, OUT_DIR, SHAPEFILE,
                    TARGET_MODEL, WORK_DIR)
from factorial import anova, contrasts
from landmask import area_weights, cached_land_mask
from preprocess import load_native

COMPONENTS = ["NOx", "VOC", "INT", "RES"]


def annual_means(model, exp):
    data, lat, lon = load_native(model, exp)
    data = data.astype(np.float64)
    annual = data.reshape(N_YEARS, 12, *data.shape[1:]).mean(axis=1) * 1e9
    return annual, lat, lon


def partition(y, lats, land):
    weights = area_weights(lats, land)
    flat = y[..., land]
    pct, _, _ = anova(flat)
    fractions = [float(np.einsum("p,p->", pct[k], weights)) for k in range(4)]
    d = contrasts(flat.mean(axis=1))[3]
    return fractions, float(np.einsum("p,p->", d, weights))


def main():
    grid = np.load(WORK_DIR / "grid.npz")
    clat, clon = grid["lats"], grid["lons"]
    cland = cached_land_mask(clat, clon, SHAPEFILE, WORK_DIR / "land_mask.npy")

    rows = []
    for model in MODELS:
        common = np.stack([np.load(WORK_DIR / f"surf_{model}_{e}_monthly.npy").astype(np.float64)
                           for e in EXPERIMENTS]).mean(axis=2) * 1e9
        cfrac, cd = partition(common, clat, cland)

        stack, lat, lon = [], None, None
        for e in EXPERIMENTS:
            a, lat, lon = annual_means(model, e)
            stack.append(a)
        nland = cached_land_mask(lat, lon, SHAPEFILE,
                                 WORK_DIR / f"land_mask_{model}.npy")
        nfrac, nd = partition(np.stack(stack), lat, nland)

        row = {"model": model, "grid": f"{len(lat)}x{len(lon)}"}
        for k, name in enumerate(COMPONENTS):
            row[f"{name}_native"], row[f"{name}_common"] = nfrac[k], cfrac[k]
        row["D_native"], row["D_common"] = nd, cd
        rows.append(row)
        print(f"  {model:12s} {row['grid']:>9s} "
              f"NOx {nfrac[0]:6.2f} / {cfrac[0]:6.2f}   D {nd:6.3f} / {cd:6.3f}")

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "regrid_sensitivity.csv", index=False)

    print(f"\nmean NOx fraction : native {table.NOx_native.mean():.2f}% "
          f"vs common {table.NOx_common.mean():.2f}% "
          f"({table.NOx_native.mean() - table.NOx_common.mean():+.2f} pp)")
    print(f"mean interaction  : native {table.INT_native.mean():.3f}% "
          f"vs common {table.INT_common.mean():.3f}% "
          f"({table.INT_native.mean() - table.INT_common.mean():+.3f} pp)")
    print(f"max |change in D| : {(table.D_native - table.D_common).abs().max():.3f} ppb")

    ranked = all(r.NOx_native == max(r[f"{c}_native"] for c in COMPONENTS)
                 for _, r in table.iterrows())
    print(f"NOx ranks first on every native grid: {ranked}")

    control = table.loc[table.model == TARGET_MODEL]
    if not control.empty:
        delta = float((control.NOx_native - control.NOx_common).abs().max())
        print(f"{TARGET_MODEL} supplies the common grid, so its difference "
              f"must be zero: {delta:.3e} pp")


if __name__ == "__main__":
    main()
