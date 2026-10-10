import json

import numpy as np
import pandas as pd
from scipy import stats

from config import EXPERIMENTS, MODELS, N_YEARS, OUT_DIR, SHAPEFILE, WORK_DIR
from factorial import H, anova, bh, bootstrap_weights, component_variance, contrasts, land_mean
from landmask import area_weights, cached_land_mask, cached_region_masks

BLOCKS = [(1, 2000), (3, 2000), (5, 4000), (7, 2000)]
RESPONSES = ["NOx main effect", "VOCs+CO main effect", "Non-additive response D"]
MAIN_BLOCK = 5


def load_inputs():
    grid = np.load(WORK_DIR / "grid.npz")
    lats, lons = grid["lats"], grid["lons"]
    land = cached_land_mask(lats, lons, SHAPEFILE, WORK_DIR / "land_mask.npy")
    weights = area_weights(lats, land)
    y = np.stack([np.stack([np.load(WORK_DIR / f"surf_{m}_{e}_monthly.npy")[..., land]
                            for e in EXPERIMENTS]) for m in MODELS]).astype(float) * 1e9
    return y, land, weights, lats, lons


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y, land, weights, lats, lons = load_inputs()
    n_point = int(land.sum())

    def wm(x, mask=None):
        return land_mean(x, weights, None if mask is None else mask[land])

    def to_map(x):
        out = np.full(x.shape[:-1] + land.shape, np.nan)
        out[..., land] = x
        return out

    annual = y.mean(axis=3)
    pct, ss, _ = anova(annual)
    mme = annual.mean(axis=0)
    mpct, mss, mfp = anova(mme)

    rows = []
    fractions = list(wm(pct)) + [wm(pct).mean(axis=0), wm(mpct)]
    names = MODELS + ["Mean of model-specific fractions", "MME field"]
    for name, p in zip(names, fractions):
        rows.append(dict(zip(["model", "NOx_pct", "VOCs_CO_pct", "interaction_pct",
                              "residual_pct"], [name] + list(p))))
    pd.DataFrame(rows).to_csv(OUT_DIR / "variance_fractions.csv", index=False)

    r = contrasts(annual.mean(axis=2))
    mr = r.mean(axis=0)
    assert np.allclose(r[:, 0], r[:, 1] + r[:, 2] + r[:, 3])
    response = wm(r)

    rows = []
    for name, v in zip(MODELS + ["MME"], list(response) + [response.mean(axis=0)]):
        row = dict(zip(["model", "total", "NOx_low", "VOCs_CO_low", "D",
                        "VOCs_CO_high", "NOx_high"], [name] + list(v)))
        row.update(VOCs_CO_ratio=v[4] / v[2], NOx_ratio=v[5] / v[1],
                   D_share_total=100 * v[3] / v[0], D_share_VOCs_CO_high=100 * v[3] / v[4])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "response_components.csv", index=False)
    print("ANOVA and response identities verified", flush=True)

    diagnostics, rhos, shaps = [], [], []
    for name, arr in zip(MODELS + ["MME"], list(annual) + [mme]):
        residual = arr - arr.mean(axis=1, keepdims=True)
        ar = residual[:, :-1] - residual[:, :-1].mean(axis=1, keepdims=True)
        br = residual[:, 1:] - residual[:, 1:].mean(axis=1, keepdims=True)
        rho = (ar * br).sum(axis=1) / np.sqrt((ar ** 2).sum(axis=1) * (br ** 2).sum(axis=1))
        shap = stats.shapiro(residual, axis=1).pvalue
        lev = stats.levene(*list(residual), center="median", axis=0).pvalue
        rhos.append(rho)
        shaps.append(shap)
        for e, rr, sp in zip(EXPERIMENTS, rho, shap):
            diagnostics.append({"model": name, "experiment": e,
                                "mean_abs_rho": wm(np.abs(rr)),
                                "area_abs_rho_gt_0_3_pct": 100 * wm(np.abs(rr) > .3),
                                "normality_rejected_area_pct": 100 * wm(sp < .05),
                                "levene_rejected_area_pct": 100 * wm(lev < .05)})
    pd.DataFrame(diagnostics).to_csv(OUT_DIR / "residual_diagnostics.csv", index=False)

    boot_summary, main_boot, model_boot_p, main_boot_response = [], None, None, None
    boot_low = boot_high = None
    for block, b in BLOCKS:
        rng = np.random.default_rng(260907 + block)
        dstar = np.zeros((b, n_point))
        sample_response = np.zeros((b, 6))
        model_ps, observed = [], mr[3]
        for im in range(len(MODELS)):
            cells = [bootstrap_weights(rng, b, N_YEARS, block) @ annual[im, ie]
                     for ie in range(len(EXPERIMENTS))]
            br = contrasts(np.stack(cells, axis=1))
            dstar += br[:, 3] / len(MODELS)
            sample_response += wm(br) / len(MODELS)
            model_ps.append((1 + (np.abs(br[:, 3] - r[im, 3]) >= np.abs(r[im, 3])).sum(axis=0))
                            / (b + 1))
            del br, cells
        p = (1 + (np.abs(dstar - observed) >= np.abs(observed)).sum(axis=0)) / (b + 1)
        q = bh(p)
        boot_summary.append({"block_years": block, "resamples": b,
                             "significant_area_pct": 100 * wm(p < .05),
                             "F_test_agreement_area_pct": 100 * wm((p < .05) == (mfp[2] < .05)),
                             "FDR_significant_area_pct": 100 * wm(q < .05),
                             "global_D_low": np.quantile(sample_response[:, 3], .025),
                             "global_D_high": np.quantile(sample_response[:, 3], .975)})
        if block == MAIN_BLOCK:
            main_boot, model_boot_p, main_boot_response = p, np.stack(model_ps), sample_response
            boot_low, boot_high = np.quantile(dstar, [.025, .975], axis=0)
        print("bootstrap block", block, boot_summary[-1], flush=True)
    pd.DataFrame(boot_summary).to_csv(OUT_DIR / "bootstrap_sensitivity.csv", index=False)

    month_clim = y.mean(axis=2)
    effect = np.einsum("ke,mesp->mksp", H, month_clim)
    vp, vterms = component_variance(effect, weights, None)

    rng = np.random.default_rng(4907)
    vb = []
    for ib in range(500):
        mc = np.empty_like(month_clim)
        for im in range(len(MODELS)):
            for ie in range(len(EXPERIMENTS)):
                bw = bootstrap_weights(rng, 1, N_YEARS, MAIN_BLOCK)[0]
                mc[im, ie] = np.einsum("y,ysp->sp", bw, y[im, ie])
        vpstar, _ = component_variance(np.einsum("ke,mesp->mksp", H, mc), weights, None)
        vb.append(vpstar)
        if (ib + 1) % 100 == 0:
            print("model-month bootstrap", ib + 1, flush=True)
    vb = np.stack(vb)
    corrected = 2 * vp - vb.mean(axis=0)
    rawlo, rawhi = np.quantile(vb, [.025, .975], axis=0)
    vlo, vhi = 2 * vp - rawhi, 2 * vp - rawlo

    vrows = []
    for k, name in enumerate(RESPONSES):
        for j, term in enumerate(["Model", "Month", "Model x month"]):
            vrows.append({"response": name, "component": term, "fraction_pct": vp[k, j],
                          "bias_corrected_pct": corrected[k, j],
                          "bootstrap_low": vlo[k, j], "bootstrap_high": vhi[k, j],
                          "variance_ppb2": wm(vterms)[k, j]})
    pd.DataFrame(vrows).to_csv(OUT_DIR / "model_month_partition.csv", index=False)

    loo_rows = []
    for drop, dropped in enumerate(MODELS):
        keep = [i for i in range(len(MODELS)) if i != drop]
        vp_loo, _ = component_variance(
            np.einsum("ke,mesp->mksp", H, month_clim[keep]), weights, None)
        for k, name in enumerate(RESPONSES):
            loo_rows.append({"omitted_model": dropped, "response": name,
                             "model_pct": vp_loo[k, 0], "month_pct": vp_loo[k, 1],
                             "model_month_pct": vp_loo[k, 2],
                             "month_plus_interaction_pct": vp_loo[k, 1] + vp_loo[k, 2]})
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(OUT_DIR / "model_month_leave_one_out.csv", index=False)
    loo_range = {}
    for k, name in enumerate(RESPONSES):
        sub = loo[loo.response == name]
        loo_range[name] = {
            "model_pct_min": float(sub.model_pct.min()),
            "model_pct_max": float(sub.model_pct.max()),
            "month_plus_interaction_min": float(sub.month_plus_interaction_pct.min()),
            "month_plus_interaction_max": float(sub.month_plus_interaction_pct.max()),
            "full_model_pct": float(vp[k, 0]),
            "full_month_plus_interaction": float(vp[k, 1] + vp[k, 2])}
        print(f"leave-one-out {name}: model {sub.model_pct.min():.1f}-{sub.model_pct.max():.1f}%, "
              f"month+interaction {sub.month_plus_interaction_pct.min():.1f}-"
              f"{sub.month_plus_interaction_pct.max():.1f}%", flush=True)

    monthly = np.moveaxis(y, 3, 1)
    month_pct, month_ss, _ = anova(monthly)
    seasons = [[11, 0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10]]
    seasonal_y = np.stack([y[:, :, :, s].mean(axis=3) for s in seasons], axis=1)
    season_pct, season_ss, _ = anova(seasonal_y)
    monthly_response = contrasts(np.moveaxis(month_clim, 2, 1))
    seasonal_response = contrasts(seasonal_y.mean(axis=3))

    regions = cached_region_masks(lats, lons, SHAPEFILE, WORK_DIR / "region_masks.npz")
    n_cell = len(EXPERIMENTS) * N_YEARS - 1
    region_rows = []
    for name in sorted(regions):
        mask = regions[name] & land
        if not mask.any():
            continue
        val = wm(mr, mask)
        pv = wm(pct.mean(axis=0), mask)
        region_rows.append(dict(
            region=name, total=val[0], NOx_low=val[1], VOCs_CO_low=val[2], D=val[3],
            VOCs_CO_high=val[4], NOx_high=val[5],
            D_share_total=100 * val[3] / val[0], D_share_VOCs_CO_high=100 * val[3] / val[4],
            mean_model_NOx_pct=pv[0], mean_model_VOCs_CO_pct=pv[1], mean_model_D_pct=pv[2],
            total_variance_MME_ppb2=wm(mss.sum(axis=0) / n_cell, mask),
            significant_area_pct=100 * wm(main_boot < .05, mask),
            model_D_min=wm(r[:, 3], mask).min(), model_D_max=wm(r[:, 3], mask).max()))
    pd.DataFrame(region_rows).to_csv(OUT_DIR / "regional_results.csv", index=False)

    np.savez_compressed(
        OUT_DIR / "analysis_fields.npz",
        mean_model_pct=to_map(pct.mean(axis=0)), model_pct=to_map(pct), mme_pct=to_map(mpct),
        response=to_map(mr), model_response=to_map(r),
        boot_p=to_map(main_boot), boot_q=to_map(bh(main_boot)),
        model_boot_p=to_map(model_boot_p),
        boot_D_low=to_map(boot_low), boot_D_high=to_map(boot_high), f_p=to_map(mfp),
        rho=to_map(np.stack(rhos)), shapiro=to_map(np.stack(shaps)),
        monthly_pct=to_map(month_pct), monthly_ss=to_map(month_ss),
        monthly_response=to_map(monthly_response), monthly_effect=to_map(effect),
        seasonal_pct=to_map(season_pct), seasonal_ss=to_map(season_ss),
        seasonal_response=to_map(seasonal_response),
        model_month_fraction=vp, model_month_corrected=corrected,
        model_month_low=vlo, model_month_high=vhi, model_month_variance=wm(vterms))

    summary = {
        "n_models": len(MODELS), "n_years": N_YEARS, "land_cells": n_point,
        "mean_model_fraction": wm(pct).mean(axis=0).tolist(),
        "MME_fraction": wm(mpct).tolist(),
        "response": response.mean(axis=0).tolist(),
        "D_positive_area_pct": float(100 * wm(mr[3] > 0)),
        "all_models_D_positive_area_pct": float(100 * wm(np.all(r[:, 3] > 0, axis=0))),
        "model_D_sign_disagreement_area_pct":
            float(100 * wm((r[:, 3].min(axis=0) < 0) & (r[:, 3].max(axis=0) > 0))),
        "bootstrap": boot_summary, "model_month_fraction": vp.tolist(),
        "model_month_corrected": corrected.tolist(),
        "model_month_leave_one_out": loo_range,
    }
    (OUT_DIR / "verified_numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
