import awkward as ak
import numpy as np
from numba import njit, prange


def softkiller_thresholds(
    vectors, eta_bins=np.linspace(-2.5, 2.5, 6), phi_bins=np.linspace(-np.pi, np.pi, 7)
):
    @njit(parallel=True)
    def event_sk(pt, eta, phi, eta_bins, phi_bins):
        local_maxima = np.zeros((len(eta_bins) - 1, (len(phi_bins) - 1)))
        for i in prange(local_maxima.shape[0]):
            eta_mask = (eta > eta_bins[i]) & (eta < eta_bins[i + 1])
            for j in prange(local_maxima.shape[1]):
                phi_mask = (phi > phi_bins[j]) & (phi < phi_bins[j + 1])
                mask = eta_mask & phi_mask
                if np.sum(mask) > 0:
                    local_maxima[i, j] = np.max(pt[mask])

        return np.median(local_maxima.flatten())

    print("Calculating SoftKiller thresholds...")

    thresholds = np.empty(len(vectors))
    for i, event in enumerate(vectors):
        thresholds[i] = event_sk(
            ak.to_numpy(event.pt),
            ak.to_numpy(event.eta),
            ak.to_numpy(event.phi),
            eta_bins=eta_bins,
            phi_bins=phi_bins,
        )

    if isinstance(vectors, ak.Array):
        thresholds = ak.from_numpy(thresholds)

    return thresholds


def apply_softkiller(vectors, **kwargs):
    thresh = softkiller_thresholds(vectors, **kwargs)
    return vectors[vectors.pt > thresh]
