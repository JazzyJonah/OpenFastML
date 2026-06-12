import numpy as np
import awkward as ak

def efex_ix(efex_vector):
    """
    Gets the indices of efex(eta,phi) with respect to a 50x64 tower window.
    """
    H = 50
    W = 64
    ETA_SCALE = 0.1
    PHI_SCALE = np.pi / 32

    eta_vals = (np.arange(H, dtype=np.float32) - (H - 1)/2) * ETA_SCALE
    phi_vals = (np.arange(W, dtype=np.float32) - (W - 1)/2) * PHI_SCALE

    eta = efex_vector.eta
    phi = efex_vector.phi

    seed_ix = []
    for i in range(len(eta)):
        idx_eta = np.abs(np.array(eta[i][:, None]) - eta_vals[None, :]).argmin(axis=1)
        idx_phi = np.abs(np.array(phi[i][:, None]) - phi_vals[None, :]).argmin(axis=1)

        seed_ix.append(np.stack([idx_eta, idx_phi], axis=-1))

    return ak.Array(seed_ix)

def eFex_slidingwindow_mask(efex_vectors):
    """
    In (eventsx50x64) tower windows, gets the indices that correspond to the (eventsxetaxphi) of efex vectors
    """
    seed_ix = efex_ix(efex_vectors)
    lengths = ak.to_numpy(ak.num(seed_ix, axis=1))
    ev_ids  = np.repeat(np.arange(len(seed_ix)), lengths)
    pairs = ak.to_numpy(ak.flatten(seed_ix, axis=1))
    if pairs.size == 0:

        empty = np.empty(0, dtype=np.int64)
        
        return np.empty(0, dtype=bool), (empty, empty, empty)
     
    e0, p0 = pairs[:, 0], pairs[:, 1]

    ok = (0 <= e0) & (e0 < 50) & (0 <= p0) & (p0 < 64)
    
    return ok, (ev_ids[ok], e0[ok], p0[ok])

