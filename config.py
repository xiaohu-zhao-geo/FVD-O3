import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("OZONE_DATA", "data"))
RAW_DIR = DATA_DIR / "cmip6"
WORK_DIR = DATA_DIR / "work"
SHAPEFILE = DATA_DIR / "regions" / "World_Regions_16.shp"
OUT_DIR = Path(os.environ.get("OZONE_OUT", "results"))

MODELS = ["GFDL-ESM4", "BCC-ESM1", "GISS-E2-1-G", "UKESM1-0-LL", "MRI-ESM2-0"]

EXPERIMENTS = ["piClim-control", "piClim-VOC", "piClim-NOx", "piClim-O3"]

VARIANTS = {
    "GFDL-ESM4": "r1i1p1f1_gr1",
    "BCC-ESM1": "r1i1p1f1_gn",
    "GISS-E2-1-G": "r1i1p3f1_gn",
    "UKESM1-0-LL": "r1i1p1f4_gn",
    "MRI-ESM2-0": "r1i1p1f1_gn",
}

MONTH_SLICE = slice(12, 360)
N_YEARS = 29

TARGET_MODEL = "BCC-ESM1"


def raw_files(model, exp):
    pattern = f"o3_AERmon_{model}_{exp}_{VARIANTS[model]}_*.nc"
    return sorted(RAW_DIR.glob(pattern))
