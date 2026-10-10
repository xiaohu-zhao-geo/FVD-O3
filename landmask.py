import numpy as np


def build_land_mask(lats, lons, shapefile):
    import geopandas as gpd
    from shapely.geometry import Point

    regions = gpd.read_file(str(shapefile)).to_crs("EPSG:4326")
    geoms = [g.buffer(0) for g in regions.geometry if g is not None and not g.is_empty]

    lon180 = np.asarray(lons, dtype=float)
    if np.nanmax(lon180) > 180:
        lon180 = np.where(lon180 > 180, lon180 - 360, lon180)

    mask = np.zeros((len(lats), len(lon180)), dtype=bool)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lon180):
            point = Point(float(lon), float(lat))
            mask[i, j] = any(g.contains(point) or g.intersects(point) for g in geoms)
    return mask


def area_weights(lats, mask):
    w = np.broadcast_to(np.cos(np.deg2rad(lats))[:, None], mask.shape)[mask].astype(float)
    return w / w.sum()


def cached_land_mask(lats, lons, shapefile, cache):
    if cache.exists():
        mask = np.load(cache)
        if mask.shape == (len(lats), len(lons)):
            return mask.astype(bool)
    mask = build_land_mask(lats, lons, shapefile)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, mask)
    return mask


def build_region_masks(lats, lons, shapefile, field="REGION_16"):
    import geopandas as gpd
    from shapely.geometry import Point

    regions = gpd.read_file(str(shapefile)).to_crs("EPSG:4326")
    lon180 = np.asarray(lons, dtype=float)
    if np.nanmax(lon180) > 180:
        lon180 = np.where(lon180 > 180, lon180 - 360, lon180)

    out = {}
    for name, group in regions.groupby(field):
        geoms = [g.buffer(0) for g in group.geometry if g is not None and not g.is_empty]
        mask = np.zeros((len(lats), len(lon180)), dtype=bool)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lon180):
                point = Point(float(lon), float(lat))
                mask[i, j] = any(g.contains(point) or g.intersects(point) for g in geoms)
        out[str(name)] = mask
    return out


def cached_region_masks(lats, lons, shapefile, cache, field="REGION_16"):
    if cache.exists():
        z = np.load(cache)
        first = z[z.files[0]]
        if first.shape == (len(lats), len(lons)):
            return {k: z[k].astype(bool) for k in z.files}
    masks = build_region_masks(lats, lons, shapefile, field)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **masks)
    return masks
