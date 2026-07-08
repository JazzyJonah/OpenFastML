import os
import ROOT
import numpy as np

N_ETA = 50
N_PHI = 64
N_CH = 2


def get_root_data(
        config: dict[str, float | int | str],
        n_start: int,
        n_stop: int,
):
    """Get an RDataFrame corresponding to one dataset.
    
    n_start, n_stop: range of 10k files loaded.
    """

    path: str = config["dir"]
    files = [
        os.path.join(path, f"events{i*10}k_{(i+1)*10}k.root")
        for i in range(n_start, n_stop)
    ]
    files_noPU = [
        os.path.join(path, f"events{i*10}k_{(i+1)*10}k_noPU.root")
        for i in range(n_start, n_stop)
    ]
    
    chain = ROOT.TChain("Delphes")
    for file in files:
        chain.Add(file)
    
    chain_noPU = ROOT.TChain("Delphes")
    for file in files_noPU:
        chain_noPU.Add(file)
    
    chain.AddFriend(chain_noPU, "noPU")

    rdf = ROOT.RDataFrame(chain)

    return rdf, chain_noPU

def add_towers(
        df,
        eta_edges=np.linspace(-2.5, 2.5, N_ETA+1),
        phi_edges=np.linspace(-np.pi, np.pi, N_PHI+1),
):
    tower_edges = (np.arange(1 + df.Count().GetValue()), eta_edges, phi_edges)
    def _histo(eta, phi, pt_eem, pt_ehad):
        out = np.zeros((N_ETA * N_PHI * 2), dtype=np.float64)

        for i in range(len(eta)):
            eta_bin = np.searchsorted(tower_edges[1], eta[i], side="right") - 1
            phi_bin = np.searchsorted(tower_edges[2], phi[i], side="right") - 1

            if eta_bin > N_ETA - 1:
                eta_bin = N_ETA - 1
            if phi_bin > N_PHI - 1:
                phi_bin = N_PHI - 1

            index = eta_bin * N_PHI + phi_bin

            out[2 * index] += pt_eem[i]
            out[2 * index + 1] += pt_ehad[i]

        return out # ORDER: [e=0,p=0,ch=0], [e=0,p=0,ch=1], [e=0, p=1, ch=0], ..., [e=49, p=63, ch=1]
    
    for prefix, suffix in [("", ""), ("noPU.", "_noPU")]:
        df = (
            df.Define(f"eta{suffix}", _mask, [f"{prefix}Tower.Eta", f"{prefix}Tower.Eta"])
            .Define(f"phi{suffix}", _mask, [f"{prefix}Tower.Phi", f"{prefix}Tower.Eta"])
            .Define(f"pt_eem{suffix}", _mask, [f"{prefix}Tower.Eem", f"{prefix}Tower.Eta"])
            .Define(f"pt_ehad{suffix}", _mask, [f"{prefix}Tower.Ehad", f"{prefix}Tower.Eta"])
            .Define(f"towers{suffix}", _histo, [f"eta{suffix}", f"phi{suffix}", f"pt_eem{suffix}", f"pt_ehad{suffix}"])
        )

    return df

def _mask(x, eta):
    return x[np.abs(eta) <= 2.5]


def add_seed_and_truth_vectors(df, sample_name):
    if 'Zee' in sample_name:
        selectCol = "towers_noPU"
    else:
        selectCol = "towers"
    df = df.Define("select_pix", _select_pix, [selectCol])
    # select_pix is at e0*64 + p0
    # e0 is at select_pix // 64
    # p0 is at select_pix % 64

    df = df.Define("select_seed_pix", _eta_pt_mask_pix, ["select_pix", "towers"])
    # df = df.Define("sum_select_seed_pt", _select_sum, ["select_seed_pix", "towers"])
    # print(df.Sum("sum_select_seed_pt").GetValue()) 
    # # Prints 305299.29713344574 for zee, which is what we want!
    # # Prints 1544621.4271774292 for jz; slightly off

    df = df.Define("dropped_seed_pix", _drop_overlapping, ["select_seed_pix", "towers"])
    
    # df = df.Define("sum_dropped_seed_pt", _select_sum, ["dropped_seed_pix", "towers"])
    # print(df.Sum("sum_dropped_seed_pt").GetValue())
    # # Prints 302164.2581586838 for zee, which is almost exactly correct 
    # # (I'll chalk it to a rounding error that was present in the 
    # # original code that could cause bad things to happen)
    # # Prints 1544621.4271774292 for jz; slightly off

    df = df.Define("dropped_seed_pt", _pt_at_pix, ["dropped_seed_pix", "towers"])

    
    if sample_name == "Zee": # I don't think this is ever used.
        df = df.Define("truth_pix", _drop_overlapping, ["select_seed_pix", "towers_noPU"])

        # df = df.Define("sum_truth_pt", _select_sum, ["truth_pix", "towers_noPU"])
        # print(df.Sum("sum_truth_pt").GetValue())
        # # Prints 295109.65280246735

        df = df.Define("truth_seed_pt", _pt_at_pix, ["truth_pix", "towers_noPU"])

    df = df.Define("seed_x_bank", _seed_x_bank, ["dropped_seed_pix", "towers"])

    return df

def _select_pix(towers):
    return np.flatnonzero(towers[0::2] > 10)


eta_edges = np.linspace(-2.5, 2.5, N_ETA+1)[:-1] # Exclude the 2.5, it's never used in the original
# NOTE: I believe in theory -2.5 should be used, but in the original implementation, it's actually cut out by the mask 
def _eta_pt_mask_pix(select_pix, towers):
    min_pt = 10
    e0 = select_pix // N_PHI
    eta = eta_edges[e0] # Equivalent to  np.tile(eta_edges, (len(self.towers), 1))[ev_ids, e0] 
    abs_eta = np.abs(eta)

    pt = towers[N_CH * select_pix]

    mask = ((abs_eta < 2.5) & (pt > min_pt) & ((abs_eta < 1.37) | (abs_eta > 1.52)))

    return select_pix[mask]

def _drop_overlapping(seed_pix, towers):
    m = len(seed_pix)

    if m == 0:
        return np.empty(0, dtype=np.int64)

    keep = np.ones(m, dtype=np.bool_)

    for i in range(m):
        pix_i = seed_pix[i]
        e_i = pix_i // 64
        p_i = pix_i % 64
        pt_i = towers[2 * pix_i]

        for j in range(m):
            if i == j:
                continue

            pix_j = seed_pix[j]
            e_j = pix_j // 64
            p_j = pix_j % 64

            if abs(e_i - e_j) > 2 or abs(p_i - p_j) > 2:
                continue

            pt_j = towers[2 * pix_j]

            if pt_j > pt_i or (pt_j == pt_i and j < i):
                keep[i] = False
                break

    n_keep = 0
    for i in range(m):
        if keep[i]:
            n_keep += 1

    out = np.empty(n_keep, dtype=np.int64)

    k = 0
    for i in range(m):
        if keep[i]:
            out[k] = seed_pix[i]
            k += 1

    return out

def _pt_at_pix(seed_pix, towers):
    return towers[N_CH * seed_pix]

SEED_SIZE = 3 * 3 * 2
def _seed_x_bank(seed_pix, towers):
    out = np.zeros(len(seed_pix) * SEED_SIZE, dtype=np.float64)

    currIndex = 0
    for pix in seed_pix:
        e0 = pix // N_PHI
        p0 = pix % N_PHI

        for dEta in range(-1, 2):
            e = e0 + dEta

            for dPhi in range(-1, 2):
                p = (p0 + dPhi) % N_PHI # Phi wraps
                if 0 <= e < N_ETA: # Eta pads with zeros, for some reason
                    towerIndex = N_CH * (e * N_PHI + p)
                    out[currIndex] = towers[towerIndex]
                    out[currIndex+1] = towers[towerIndex + 1]
                currIndex += 2 # In case the above if statement is false, zero pad
    
    return out




# For debugging only pretty much
def _select_sum(select_pix, towers):
    return np.sum(towers[2 * select_pix])

if __name__ == "__main__":
    from fastml.utils.misc import get_config
    config = get_config("Zee")
    rdf = get_root_data(config, 0, 2, no_PU=False)
    print(rdf.Describe())