import numpy as np
from scipy import stats

H = np.array([[-1, -1, 1, 1],
              [-1, 1, -1, 1],
              [1, -1, -1, 1]], dtype=float)
H[:2] /= 2


def anova(y):
    n = y.shape[-2]
    mu = y.mean(axis=-2)
    eff = np.einsum("ke,...ep->...kp", H, mu)
    ss = n * eff ** 2
    ss[..., 2, :] /= 4
    residual = ((y - mu[..., None, :]) ** 2).sum(axis=(-3, -2))
    ss = np.concatenate([ss, residual[..., None, :]], axis=-2)

    pct = 100 * ss / ss.sum(axis=-2, keepdims=True)
    total = ((y - y.mean(axis=(-3, -2), keepdims=True)) ** 2).sum(axis=(-3, -2))
    assert np.allclose(ss.sum(axis=-2), total, rtol=1e-10, atol=1e-9)

    f = ss[..., :3, :] / (ss[..., -1:, :] / (4 * (n - 1)))
    return pct, ss, stats.f.sf(f, 1, 4 * (n - 1))


def contrasts(mu):
    c, v, n, o = np.moveaxis(mu, -2, 0)
    return np.stack([o - c, n - c, v - c, o - n - v + c, o - n, o - v], axis=-2)


def bootstrap_weights(rng, b, n, block):
    starts = rng.integers(n, size=(b, int(np.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(b, -1)[:, :n]
    count = np.zeros((b, n))
    np.add.at(count, (np.repeat(np.arange(b), n), idx.ravel()), 1 / n)
    return count


def bh(p):
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def component_variance(effect, weights, mask):
    grand = effect.mean(axis=(0, 2), keepdims=True)
    model = effect.mean(axis=2, keepdims=True) - grand
    month = effect.mean(axis=0, keepdims=True) - grand
    cross = effect - grand - model - month

    terms = np.stack([(model ** 2).mean(axis=(0, 2)),
                      (month ** 2).mean(axis=(0, 2)),
                      (cross ** 2).mean(axis=(0, 2))], axis=1)
    total = ((effect - grand) ** 2).mean(axis=(0, 2))
    assert np.allclose(terms.sum(axis=1), total, atol=1e-10)

    integrated = land_mean(terms, weights, mask)
    return 100 * integrated / integrated.sum(axis=1, keepdims=True), terms


def land_mean(x, weights, mask=None):
    w = weights if mask is None else weights * mask
    return np.einsum("...p,p->...", x, w / w.sum())
