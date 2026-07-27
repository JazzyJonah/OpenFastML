import numpy as np
import awkward as ak
import json
from fastml.utils_egamma.efex import eFex_slidingwindow_mask
from fastml.utils.image import sliding_window

eta_edges=np.linspace(-2.5, 2.5, 51)
phi_edges=np.linspace(-np.pi, np.pi, 65)
n_eta = len(eta_edges) - 1
n_phi = len(phi_edges) - 1

def get_config(sample):
    return json.load(open("Original/Preprocessing/configs/samples.json"))[sample]

def get_jagged_towers(towers, fex, ev_ids, ok):
    n_per_evt = ak.to_numpy(ak.num(fex, axis=1))
    total_seeds = int(n_per_evt.sum())

    offsets = np.concatenate(([0], np.cumsum(n_per_evt[:-1])))
    local_idx_all = ak.to_numpy(ak.flatten(ak.local_index(fex, axis=1), axis=1))

    local_idx_ok = local_idx_all[ok]
    flat_idx_ok = offsets[ev_ids] + local_idx_ok
    seed_shape = towers.shape[1:]

    all_towers = np.full((total_seeds,)+seed_shape, np.nan, dtype=np.float32)
    all_towers[flat_idx_ok, ...] = towers
    towers_jagged = ak.unflatten(all_towers, n_per_evt)

    return towers_jagged

def get_tower_map(n_events):

    towers = np.zeros((n_events, n_eta, n_phi, 6), dtype=np.float32)

    return towers

def drop_overlapping(fex):
    ok, (ev_ids, e0, p0) = eFex_slidingwindow_mask(fex)
    towers = get_tower_map(len(fex))
    pattern = np.arange(1, n_eta * n_phi + 1, dtype=int).reshape(n_eta, n_phi)
    towers[..., 0] = pattern
    tower_windows = sliding_window(towers, 3)[ev_ids, e0, p0]
    towers_jagged = get_jagged_towers(tower_windows, fex, ev_ids, ok)
    t9 = ak.flatten(towers_jagged[..., 0], axis=3)

    drop_masks = []

    for i in range(len(t9)):
        A = np.asarray(t9[i])        
        pt = np.asarray(fex.pt[i]) 
        m = A.shape[0]

        if m == 0:
            drop_masks.append(np.zeros(0, dtype=bool))
            continue

        flat_nz = A[A != 0].ravel()
        if flat_nz.size == 0:
            drop_masks.append(np.zeros(m, dtype=bool))
            continue

        vals, counts = np.unique(flat_nz, return_counts=True)
        repeated = vals[counts > 1]
        if repeated.size == 0:
            drop_masks.append(np.zeros(m, dtype=bool))
            continue

        drop = np.zeros(m, dtype=bool)

        for tid in repeated:
            seeds_with_tid = np.any(A == tid, axis=1)
            idx = np.nonzero(seeds_with_tid)[0]
            if idx.size <= 1:
                continue

            winner = idx[np.argmax(pt[idx])]

            losers = idx[idx != winner]
            drop[losers] = True

        drop_masks.append(drop)

    return ~ak.Array(drop_masks)