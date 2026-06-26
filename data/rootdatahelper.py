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

    weightSum = rdf.Sum("Event.Weight").GetValue() # Only reweigh the non noPU weights
    def _reweight(weight):
        return weight[0] * 1 * 1 / weightSum

    rdf = rdf.Define("EventWeight", 
                    _reweight,
                    ["Event.Weight"]
                )
    return rdf

def add_towers(
        df,
        eta_edges=np.linspace(-2.5, 2.5, 51),
        phi_edges=np.linspace(-np.pi, np.pi, 65),
):
    for prefix in ["" "noPU."]:
        df = (
            df.Define(f"{prefix}eta", _mask, [f"{prefix}Tower.Eta", f"{prefix}Tower.Eta"])
            .Define(f"{prefix}phi", _mask, [f"{prefix}Tower.Phi", f"{prefix}Tower.Eta"])
            .Define(f"{prefix}pt_eem", _mask, [f"{prefix}Tower.Eem", f"{prefix}Tower.Eta"])
            .Define(f"{prefix}pt_ehad", _mask, [f"{prefix}Tower.Ehad", f"{prefix}Tower.Eta"])
        )

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

    df = df.Define("towers_noPU", _histo, ["noPU.eta", "noPU.phi", "noPU.pt_eem", "noPU.pt_ehad"])
    df = df.Define("towers", _histo, ["eta", "phi", "pt_eem", "pt_ehad"])
    # towerArray = df.AsNumpy(columns=["tower1"])["tower1"] # Shape 20k, 6400
    return df

def _mask(x, eta):
    return x[np.abs(eta) <= 2.5]
def _indices(x, entry):
    return np.full(len(x), entry, dtype=np.int64)


def add_seed_vectors(df, sample_name):
    if 'Zee' in sample_name:
        selectCol = "towers_noPU"
    else:
        selectCol = "towers"
    print(df.GetColumnType("towers_noPU")) # ROOT::VecOps::RVec<double>
    df = df.Define("select_pix", _select_pix, [selectCol])
    # select_pix is at e0*64 + p0
    # e0 is at select_pix // 64
    # p0 is at select_pix % 64
    select_pix_np = df.AsNumpy(columns=["select_pix"])["select_pix"]
    print(select_pix_np, select_pix_np.shape, np.unique(select_pix_np), np.unique(select_pix_np).shape)

N_E = 50
N_P = 64
N_PIX = N_E * N_P

def _select_pix(towers):
    n = 0
    for pix in range(50 * 64):
        if towers[2 * pix] > 10:
            n += 1

    out = np.empty(n, dtype=np.int64)

    j = 0
    for pix in range(50 * 64):
        if towers[2 * pix] > 10:
            out[j] = pix
            j += 1

    return out


def add_truth_vectors(df):
    pass

if __name__ == "__main__":
    from fastml.utils.misc import get_config
    config = get_config("Zee")
    rdf = get_root_data(config, 0, 2, no_PU=False)
    print(rdf.Describe())