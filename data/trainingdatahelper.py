import numpy as np
import awkward as ak
from scipy.interpolate import PchipInterpolator

def filter_training(loaders, towers_dict, thresholds):
    samples = ['Zee', 
               'JZ'
               ]
    
    idx_in_bin_all = []

    zee_counts = []
    bkg_avail_counts = []

    for (lo, hi) in thresholds:
        idx_in_bin = {}
        for s in samples:
            items = loaders[s].seed_vectors.pt
            if hi == thresholds[-1][1]:
                mask = (items >= lo) & (items <= hi)
            else:
                mask = (items >= lo) & (items < hi)
            idx_in_bin[s] = np.flatnonzero(mask)

        idx_in_bin_all.append(idx_in_bin)

        zee_counts.append(len(idx_in_bin['Zee']))
        bkg_avail_counts.append(len(idx_in_bin['JZ']))

    zee_counts = np.array(zee_counts, dtype=int)
    bkg_avail_counts = np.array(bkg_avail_counts, dtype=int)

    raw_target = np.minimum(zee_counts, bkg_avail_counts)

    cap_factor = 1  
    window = 4    

    smoothed_target = raw_target.copy()

    for i in range(len(raw_target)):
        lo_i = max(0, i - window)
        hi_i = min(len(raw_target), i + window + 1)
        neigh = raw_target[lo_i:hi_i]

        if len(neigh) > 1:
            neigh_mean = (neigh.sum() - raw_target[i]) / (len(neigh) - 1)
        else:
            neigh_mean = raw_target[i]

        cap = int(np.floor(cap_factor * neigh_mean))
        smoothed_target[i] = min(raw_target[i], cap)

    print(smoothed_target)
    exit()

    rng = np.random.default_rng(123) 

    training_pt_dist_lists = {s: [] for s in samples}
    training_tower_dist_lists = {s: [] for s in samples}

    for i, (lo, hi) in enumerate(thresholds):
        target_n = smoothed_target[i]
        if target_n <= 0:
            continue

        idx_in_bin = idx_in_bin_all[i]

        zee_idx = idx_in_bin['Zee']
        zee_sel = rng.choice(zee_idx, size=target_n, replace=False)
        training_pt_dist_lists['Zee'].append(loaders['Zee'].seed_vectors.pt[zee_sel])
        training_tower_dist_lists['Zee'].append(towers_dict['Zee'][zee_sel])

        remaining = target_n
        bkg_idx = idx_in_bin['JZ']
        take = min(remaining, len(bkg_idx))
        if take <= 0:
            continue
        bkg_sel = rng.choice(bkg_idx, size=take, replace=False)
        training_pt_dist_lists['JZ'].append(loaders['JZ'].seed_vectors.pt[bkg_sel])
        training_tower_dist_lists['JZ'].append(towers_dict['JZ'][bkg_sel])

        remaining -= take
        if remaining == 0:
            continue

    training_pt_dist = {
        s: (np.concatenate(training_pt_dist_lists[s]) if training_pt_dist_lists[s] else np.array([]))
        for s in samples
    }

    training_tower_dist = {
        s: (np.concatenate(training_tower_dist_lists[s], axis=0) if training_tower_dist_lists[s]
            else np.empty((0, 3, 3, 6)))
        for s in samples
    }

    return training_tower_dist, training_pt_dist

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