import os
import ROOT
import numpy as np


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

    weightSum = rdf.Sum("Event.Weight").GetValue() # Only reweight the non noPU weights
    def _reweight(weight):
        return weight[0] * 1 * 1 / weightSum

    rdf = rdf.Define("EventWeight", 
                    _reweight,
                    ["Event.Weight"]
                )
    return rdf, chain_noPU

def add_towers(
        df,
        eta_edges=np.linspace(-2.5, 2.5, 51),
        phi_edges=np.linspace(-np.pi, np.pi, 65),
):
    tower_edges = (np.arange(1 + df.Count().GetValue()), eta_edges, phi_edges)
    def _histo(eta, phi, pt_eem, pt_ehad):
        n_eta = len(tower_edges[1])-1
        n_phi = len(tower_edges[2])-1 
        out = np.zeros((n_eta * n_phi * 2), dtype=np.float64)

        for i in range(len(eta)):
            eta_bin = np.searchsorted(tower_edges[1], eta[i], side="right") - 1
            phi_bin = np.searchsorted(tower_edges[2], phi[i], side="right") - 1

            index = eta_bin * n_phi + phi_bin

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

    # towerArray = df.AsNumpy(columns=["towers_noPU"])["towers_noPU"] # Shape 20k, 6400

    return df

def _mask(x, eta):
    return x[np.abs(eta) <= 2.5]


def add_seed_vectors(df, sample_name):
    if 'Zee' in sample_name:
        selectCol = "towers_noPU"
    else:
        selectCol = "towers"
    df = df.Define("select_pix", _select_pix, [selectCol])
    # select_pix is at e0*64 + p0
    # e0 is at select_pix // 64
    # p0 is at select_pix % 64

    df = df.Define("select_seeds", _eta_pt_mask_pix, ["select_pix", "towers"])


    # def _vec_sum(x):
    #     return np.float64(np.sum(x))
    # def _pt_at_pix(pix, pt_source):
    #     # pix is e0 * 64 + p0
    #     # pt channel is ch=0, so flat index is 2 * pix
    #     return pt_source[2 * pix]
    # df = (
    # df.Define("select_seed_pt", _pt_at_pix, ["select_seeds", "towers"])
    #     .Define("sum_select_seed_pt", _vec_sum, ["select_seed_pt"])
    # )
    # print(df.Sum("sum_select_seed_pt").GetValue()) # 305299.29713344574



def _select_pix(towers):
    return np.flatnonzero(towers[0::2] > 10)

eta_edges = np.linspace(-2.5, 2.5, 51)[:-1] # Exclude the 2.5, it's never used in the original
# NOTE: I believe in theory -2.5 should be used, but in the original implementation, it's actually cut out by the mask 
def _eta_pt_mask_pix(select_pix, towers):
    min_pt = 10
    e0 = select_pix // 64
    eta = eta_edges[e0] # Equivalent to  np.tile(eta_edges, (len(self.towers), 1))[ev_ids, e0] 
    abs_eta = np.abs(eta)

    pt = towers[2 * select_pix]

    mask = ((abs_eta < 2.5) & (pt > min_pt) & ((abs_eta < 1.37) | (abs_eta > 1.52)))

    return select_pix[mask]






def add_truth_vectors(df):
    pass

if __name__ == "__main__":
    from fastml.utils.misc import get_config
    config = get_config("Zee")
    rdf = get_root_data(config, 0, 2, no_PU=False)
    print(rdf.Describe())