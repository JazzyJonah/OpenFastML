import awkward as ak
import numpy as np
from scipy.interpolate import PchipInterpolator

np.random.seed(42)

def _convert_loader_to_awkward(df) -> None:
    columns = (
        "dropped_seed_pix",
        "dropped_seed_pt",
        "seed_x_bank"
    )
    return ak.from_rdataframe(
        df,
        columns=columns,
        keep_order=True
    )

def _rechunk_seed_x(arr: ak.Array) -> ak.Array:
    seed_x = ak.unflatten(arr.seed_x_bank, 3 * 3 * 2, axis=1)
    # seed_x = ak.to_regular(seed_x, axis=2)

    return ak.with_field(arr, seed_x, "seed_x")

def _group_sample_into_threshold_bins(arr: ak.Array, thresholds: list[tuple[float]]) -> ak.Array:
    n_bins = len(thresholds)

    # Flatten only the event dimension.
    # Keep seed_x as seed -> 18.
    pt_flat = ak.to_numpy(ak.flatten(arr.dropped_seed_pt, axis=1))
    pix_flat = ak.to_numpy(ak.flatten(arr.dropped_seed_pix, axis=1))
    x_flat = ak.flatten(arr.seed_x, axis=1)

    # Build threshold edges from [(lo0, hi0), (lo1, hi1), ...]
    edges = np.asarray(
        [thresholds[0][0]] + [hi for _, hi in thresholds],
        dtype=np.float64,
    )

    # Same convention as original:
    # normal bins: [lo, hi)
    # final bin:  [lo, hi]
    bin_id = np.searchsorted(edges, pt_flat, side="right") - 1

    # Include exact final upper edge in final bin.
    bin_id[pt_flat == edges[-1]] = n_bins - 1

    valid = (bin_id >= 0) & (bin_id < n_bins)

    pt_valid = pt_flat[valid]
    pix_valid = pix_flat[valid]
    x_valid = x_flat[valid]
    bin_valid = bin_id[valid]

    # Group by threshold bin.
    order = np.argsort(bin_valid, kind="stable")

    pt_sorted = pt_valid[order]
    pix_sorted = pix_valid[order]
    x_sorted = x_valid[order]
    bin_sorted = bin_valid[order]

    counts = np.bincount(bin_sorted, minlength=n_bins)

    binned_pt = ak.unflatten(pt_sorted, counts)
    binned_pix = ak.unflatten(pix_sorted, counts)
    binned_seeds = ak.unflatten(x_sorted, counts)

    return ak.zip(
        {
            "seeds": binned_seeds,
            "pt": binned_pt,
            "pix": binned_pix,
        },
        depth_limit=1,
    )


def add_signal_background_counts(df):
    return (df.Define("n_signal", _n_seeds, ["signal_seeds"])
            .Define("n_background", _n_seeds, ["background_seeds"]))
def _n_seeds(seeds):
    return len(seeds) // 18

def add_smooth_raw_targets(df):
    df = df.Define("raw_target", _min, ["n_signal", "n_background"])
    
    means = np.asarray(_make_means(df), dtype=np.float64)
    def _make_smoothed_target(raw_target, entry):
        cap = int(means[entry])
        return min(raw_target, cap)
    
    df = df.Define("bin_idx", "static_cast<int>(rdfentry_)")
    df = df.Define("smoothed_target", _make_smoothed_target, ["raw_target", "bin_idx"])
    
    return df
def _min(n_1, n_2):
    return min(n_1, n_2)
def _make_means(df, n_bins=60, window=4):
    sums = []
    for i in range(n_bins):
        low = max(0, i - window)
        high = min(n_bins, i + window + 1)
        sums.append((df.Range(low, i).Sum("raw_target"), 
                      df.Range(i + 1, high).Sum("raw_target"),
                      high - low - 1))
    return [(i[0].GetValue()+i[1].GetValue()) / i[2] for i in sums]

def add_final_seeds(df):
    df = (df
        .Define("signal_keep_idx", _pick_random_indices,
                ["signal_pt", "smoothed_target"])
        .Define("background_keep_idx", _pick_random_indices,
                ["background_pt", "smoothed_target"])

        .Define("final_signal_pt", _take_pts,
                ["signal_pt", "signal_keep_idx"])
        .Define("final_background_pt", _take_pts,
                ["background_pt", "background_keep_idx"])
        
        .Define("final_signal_seeds", _take_seeds,
                ["signal_seeds", "signal_keep_idx"])
        .Define("final_background_seeds", _take_seeds,
                ["background_seeds", "background_keep_idx"])
    )
    return df
def _pick_random_indices(pt, target):
    return np.random.choice(np.arange(len(pt)), target, replace=False)
def _take_pts(pts, idx):
    return pts[idx]
def _take_seeds(seeds, idx):
    seed_idx = (idx[:, None] * 18 + np.arange(18)).ravel()
    return seeds[seed_idx]

def calculate_weights(X, bin_edges):
    xk = np.array([10, 12.5, 15, 17.5, 20, 25, 30, 35, 40], dtype=float)
    yk = np.array([0.3, 0.12, 0.06, 0.03, 0.015, 0.004, 0.004, 0.004, 0.004], dtype=float)

    w0_spline = PchipInterpolator(xk, yk, extrapolate=False)

    def w0(x):
        x = np.asarray(x)
        y = w0_spline(x)
        y = np.where(np.isfinite(y), y, 0.0)
        return np.clip(y, 0.0, None)

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    I_bin = w0(centers)

    X = np.asarray(X)
    base = w0(X)                          

    sum_base, _ = np.histogram(X, bins=bin_edges, weights=base)
    c = I_bin / (sum_base + 1e-12)

    b = np.digitize(X, bin_edges) - 1
    w = np.zeros_like(base)
    valid = (b >= 0) & (b < len(c))
    w[valid] = base[valid] * c[b[valid]]

    w /= np.mean(w)
    return w