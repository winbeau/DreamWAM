"""Explicit, equal-budget offline selectors. No outcome labels or implicit fusion."""

import numpy as np


def combine_scores(components, weights):
    """Mean-normalized nonnegative proxies with externally declared weights."""
    if set(components) != set(weights) or not components:
        raise ValueError("declare exactly one weight per component")
    if any(not np.isfinite(w) or w < 0 for w in weights.values()) or not any(weights.values()):
        raise ValueError("require finite nonnegative weights, at least one positive")
    shape = next(iter(components.values())).shape
    result = np.zeros(shape, dtype=np.float64)
    for name, values in components.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("incompatible or invalid component scores")
        result += weights[name] * values / max(values.mean(), 1e-30)
    return result


def select_tokens(grid, count, *, method, scores=None, scope="all", seed=42, bottom=False):
    """Exact balanced per-frame count; fixed random seed, stable ties.

    Uniform reproduces hybrid's integer-spaced anchor positions. Camera quotas
    are deliberately not added: that would change the inherited baseline.
    """
    if scope not in ("all", "future"):
        raise ValueError("scope must be all or future")
    first_frame = 1 if scope == "future" else 0
    frames = grid.frames - first_frame
    if frames < 1 or type(count) is not int or not frames <= count <= frames * grid.frame_size:
        raise ValueError("budget must cover all included frames without overflow")
    if method not in ("uniform", "random", "score"):
        raise ValueError("selector must be uniform, random or explicit score")
    if method == "score":
        scores = np.asarray(scores)
        if scores.shape != (grid.length,) or not np.isfinite(scores).all():
            raise ValueError("scores must be a finite full-grid vector")
    elif scores is not None or bottom:
        raise ValueError("uniform/random selectors do not consume score options")
    generator = np.random.default_rng(seed)
    result = []
    for frame in range(frames):
        quota = count // frames + int(frame < count % frames)
        offset = (first_frame + frame) * grid.frame_size
        if method == "uniform":
            local = np.arange(quota) * grid.frame_size // quota
        elif method == "random":
            local = generator.permutation(grid.frame_size)[:quota]
        else:
            values = scores[offset:offset + grid.frame_size]
            local = np.argsort(values if bottom else -values, kind="stable")[:quota]
        result.extend((local + offset).tolist())
    return np.array(sorted(result), dtype=np.int64)
