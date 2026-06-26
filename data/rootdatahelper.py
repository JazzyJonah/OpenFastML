import os
import ROOT
import numpy as np


def get_root_data(
        config: dict[str, float | int | str],
        n_start: int,
        n_stop: int,
        no_PU: bool=False,
        **kwargs
):
    """Get an RDataFrame corresponding to one dataset.
    
    n_start, n_stop: range of 10k files loaded.
    """

    path: str = config["dir"]
    if no_PU:
        files = [
            os.path.join(path, f"events{i*10}k_{(i+1)*10}k_noPU.root")
            for i in range(n_start, n_stop)
        ]
    else:
        files = [
            os.path.join(path, f"events{i*10}k_{(i+1)*10}k.root")
            for i in range(n_start, n_stop)
        ]

    rdf = ROOT.RDataFrame("Delphes", set(files))

    weightSum = rdf.Sum("Event.Weight").GetValue()
    def _reweight(weight):
        return weight[0] * 1 * 1 / weightSum

    rdf = rdf.Define("EventWeight", 
                    _reweight,
                    ["Event.Weight"]
                )
    return rdf

def add_tower(
        df,
        noPU: bool,
        eta_edges=np.linspace(-2.5, 2.5, 51),
        phi_edges=np.linspace(-np.pi, np.pi, 65),
):
    tower_edges = (np.arange(1 + df.Count().GetValue()), eta_edges, phi_edges)
    df = (
        df.Define("eta", _mask, ["Tower.Eta", "Tower.Eta"])
        .Define("phi", _mask, ["Tower.Phi", "Tower.Eta"])
        .Define("pt_eem", _mask, ["Tower.Eem", "Tower.Eta"])
        .Define("pt_ehad", _mask, ["Tower.Ehad", "Tower.Eta"])
    )

    df = df.Define("entry_i64", "static_cast<Long64_t>(rdfentry_)")
    df = df.Define("event_indices", _indices, ["eta", "entry_i64"])

    # nbins = ROOT.std.vector("int")()
    # xbins = ROOT.std.vector("std::vector<double>")()
    # for edge in tower_edges:
    #     nbins.push_back(len(edge)-1)
    #     edge_vec = ROOT.std.vector("double")()
    #     for e in edge:
    #         edge_vec.push_back(float(e))
    #     xbins.push_back(edge_vec)
    # model = ROOT.RDF.THnDModel(
    #     "_name", "_title", 3,
    #     nbins, xbins
    # )
    # h = df.HistoND(model, ["event_indices", "eta", "phi"], "pt_eem")
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

        return out
    if noPU:
        df = df.Define("towers_noPU", _histo, ["eta", "phi", "pt_eem", "pt_ehad"])
    else:
        df = df.Define("towers", _histo, ["eta", "phi", "pt_eem", "pt_ehad"])
    # towerArray = df.AsNumpy(columns=["tower1"])["tower1"] # Shape 20k, 6400
    return df

def _mask(x, eta):
    return x[np.abs(eta) <= 2.5]
def _indices(x, entry):
    return np.full(len(x), entry, dtype=np.int64)



if __name__ == "__main__":
    from fastml.utils.misc import get_config
    config = get_config("Zee")
    rdf = get_root_data(config, 0, 2, no_PU=False)
    print(rdf.Describe())