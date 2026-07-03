import os
import numpy as np
import ROOT
from fastml.utils.misc import get_config

rds_bin_edges = np.arange(10.0, 40.0 + 0.5, 0.5)
rds_thresholds = [(i, i + 0.5) for i in np.arange(10, 40, 0.5)]

samples = ["Zee", "JZ"]
rds_loaders: dict[str, dict] = {}

sample_name = samples[0]

def get_root_data(config, n_start, n_stop):
    path = config["dir"]
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
    weightSum = rdf.Sum("Event.Weight").GetValue()


    def _reweight(weight):
        return weight[0] * 1 * 1 / weightSum

    rdf = rdf.Define("EventWeight", _reweight, ["Event.Weight"])
    return rdf, chain_noPU

def _mask(x, eta):
    return x[np.abs(eta) <= 2.5]


def add_towers(df, _temp, eta_edges=np.linspace(-2.5, 2.5, 51), phi_edges=np.linspace(-np.pi, np.pi, 65)):
    tower_edges = (np.arange(1 + df.Count().GetValue()), eta_edges, phi_edges)

    def _histo(eta, phi, pt_eem, pt_ehad):
        n_eta = len(tower_edges[1]) - 1
        n_phi = len(tower_edges[2]) - 1
        out = np.zeros((n_eta * n_phi * 2), dtype=np.float64)

        for i in range(len(eta)):
            eta_bin = np.searchsorted(tower_edges[1], eta[i], side="right") - 1
            if eta_bin >= 50:
                eta_bin = 49
            phi_bin = np.searchsorted(tower_edges[2], phi[i], side="right") - 1
            if phi_bin >= 64:
                phi_bin = 64

            index = eta_bin * n_phi + phi_bin

            out[2 * index] += pt_eem[i]
            out[2 * index + 1] += pt_ehad[i]

        return out

    
    for prefix, suffix in [("", ""), ("noPU.", "_noPU")]:
        df = (
            df.Define(f"eta{suffix}", _mask, [f"{prefix}Tower.Eta", f"{prefix}Tower.Eta"])
            .Define(f"phi{suffix}", _mask, [f"{prefix}Tower.Phi", f"{prefix}Tower.Eta"])
            .Define(f"pt_eem{suffix}", _mask, [f"{prefix}Tower.Eem", f"{prefix}Tower.Eta"])
            .Define(f"pt_ehad{suffix}", _mask, [f"{prefix}Tower.Ehad", f"{prefix}Tower.Eta"])
            # .Define(
            #     f"towers{suffix}",
            #     f"""auto out = std::vector<double>(50*64*2, 0.0); 
            #       const double pi = 3.141592653589793; 
            #       for (unsigned i = 0; i < eta{suffix}.size(); ++i) {{
            #           int eta_bin = int((eta{suffix}[i] + 2.5) / 0.1); 
            #           if (eta_bin < 0) continue; 
            #           if (eta_bin >= 50) eta_bin = 49; 
            #           int phi_bin = int((phi{suffix}[i] + pi) / (2*pi/64.0)); 
            #           if (phi_bin < 0) continue; 
            #           if (phi_bin >= 64) phi_bin = 63; 
            #           unsigned index = eta_bin * 64 + phi_bin; 
            #           out[2 * index] += pt_eem{suffix}[i]; 
            #           out[2 * index + 1] += pt_ehad{suffix}[i]; 
            #         }} 
            #         return out;"""
            # )
            .Define(
                f"towers{suffix}", _histo, [f"eta{suffix}", f"phi{suffix}", f"pt_eem{suffix}", f"pt_ehad{suffix}"]
            )
        )
    
    return df, _temp


def add_seed_and_truth_vectors(df, _temp, sample_name, thresholds):
    select_col = "towers_noPU" if "Zee" in sample_name else "towers"

    df = df.Define(
        "select_pix",
        "auto out = std::vector<long>(); for (unsigned i = 0; i < "
        + f"{select_col}.size() / 2; ++i) {{ if ({select_col}[2 * i] > 10) out.push_back(i); }} return out;"
    )

    df = df.Define(
        "select_seed_pix",
        "auto out = std::vector<long>(); for (unsigned i = 0; i < select_pix.size(); ++i) { long seed = select_pix[i]; long e0 = seed / 64; double eta = -2.5 + 0.1 * e0; double abs_eta = std::abs(eta); double pt = towers[2 * seed]; if ((abs_eta < 2.5) && (pt > 10) && ((abs_eta < 1.37) || (abs_eta > 1.52))) out.push_back(seed); } return out;"
    )

   
    df = df.Define(
        "dropped_seed_pix",
        "auto m = select_seed_pix.size(); if (m == 0) return std::vector<long>(); auto keep = std::vector<unsigned char>(m, 1); for (unsigned i = 0; i < m; ++i) { long pix_i = select_seed_pix[i]; long e_i = pix_i / 64; long p_i = pix_i % 64; double pt_i = towers[2 * pix_i]; for (unsigned j = 0; j < m; ++j) { if (i == j) continue; long pix_j = select_seed_pix[j]; long e_j = pix_j / 64; long p_j = pix_j % 64; if (std::abs(e_i - e_j) > 2 || std::abs(p_i - p_j) > 2) continue; double pt_j = towers[2 * pix_j]; if (pt_j > pt_i || (pt_j == pt_i && j < i)) { keep[i] = 0; break; } } } int n_keep = 0; for (unsigned i = 0; i < m; ++i) if (keep[i]) ++n_keep; auto out = std::vector<long>(); out.reserve(n_keep); for (unsigned i = 0; i < m; ++i) if (keep[i]) out.push_back(select_seed_pix[i]); return out;"
    )


    df = df.Define(
        "dropped_seed_pt",
        "auto out = std::vector<double>(); out.reserve(dropped_seed_pix.size()); for (unsigned i = 0; i < dropped_seed_pix.size(); ++i) out.push_back(towers[2 * dropped_seed_pix[i]]); return out;"
    )

    if sample_name == "Zee":
        df = df.Define(
            "truth_pix",
            "auto m = select_seed_pix.size(); if (m == 0) return std::vector<long>(); auto keep = std::vector<unsigned char>(m, 1); for (unsigned i = 0; i < m; ++i) { long pix_i = select_seed_pix[i]; long e_i = pix_i / 64; long p_i = pix_i % 64; double pt_i = towers_noPU[2 * pix_i]; for (unsigned j = 0; j < m; ++j) { if (i == j) continue; long pix_j = select_seed_pix[j]; long e_j = pix_j / 64; long p_j = pix_j % 64; if (std::abs(e_i - e_j) > 2 || std::abs(p_i - p_j) > 2) continue; double pt_j = towers_noPU[2 * pix_j]; if (pt_j > pt_i || (pt_j == pt_i && j < i)) { keep[i] = 0; break; } } } int n_keep = 0; for (unsigned i = 0; i < m; ++i) if (keep[i]) ++n_keep; auto out = std::vector<long>(); out.reserve(n_keep); for (unsigned i = 0; i < m; ++i) if (keep[i]) out.push_back(select_seed_pix[i]); return out;"
        )
        df = df.Define(
            "truth_seed_pt",
            "auto out = std::vector<double>(); out.reserve(truth_pix.size()); for (unsigned i = 0; i < truth_pix.size(); ++i) out.push_back(towers_noPU[2 * truth_pix[i]]); return out;"
        )


    df = df.Define(
        "seed_bin_counts",
        "auto counts = std::vector<long>(60, 0); for (auto v : dropped_seed_pt) { for (int b = 0; b < 60; ++b) { double lo = 10.0 + 0.5 * b; double hi = lo + 0.5; bool in_bin = (b == 59) ? (v >= lo && v <= hi) : (v >= lo && v < hi); if (in_bin) { counts[b] += 1; break; } } } return counts;"
    )
    return df, _temp


if __name__ == "__main__":
    for sample_name in samples:
        stop = 2 if sample_name == "Zee" else 1
        config = get_config(sample_name)

        df, _temp = get_root_data(config, 0, stop)
        df, _temp = add_towers(df, _temp)
        df, _temp = add_seed_and_truth_vectors(df, _temp, sample_name, rds_thresholds)

        rds_loaders[sample_name] = {"df": df, "_temp": _temp}
    print("HERE")
    s = "JZ"
    df = rds_loaders[s]["df"]

    # df.Describe().Print()
    # exit()


    print(s, "noPU tower sum", df.Sum("towers_noPU").GetValue())
    print(s, "tower sum", df.Sum("towers").GetValue())
    print(s, "select pix sum", df.Sum("select_pix").GetValue())    
    print(s, "select seed pix sum", df.Sum("select_seed_pix").GetValue())
    print(s, "dropped pix sum", df.Sum("dropped_seed_pix").GetValue())
    print(s, "dropped pt sum", df.Sum("dropped_seed_pt").GetValue())
    print(s, "seed bin counts sum", df.Sum("seed_bin_counts").GetValue())
    # df.Display(["select_pix"]).Print()
    # seed_bin_numpy = df.AsNumpy(columns=["seed_bin_counts"])["seed_bin_counts"]
    # print(seed_bin_numpy)
    # print(seed_bin_numpy.shape)
    # print(seed_bin_numpy[-1])




    if False:
        pass
        # df = df.Define("n_dropped_seed_pix", "dropped_seed_pix.size()")
        # print("Completed first def")
        # df = df.Define("sum_dropped_seed_pt", "std::accumulate(dropped_seed_pt.begin(), dropped_seed_pt.end(), 0.)")
        # print("Second")
        # df = df.Define("min_dropped_seed_pt", "auto it = std::min_element(dropped_seed_pt.begin(), dropped_seed_pt.end()); if(it != dropped_seed_pt.end()){return *it;}else{return std::numeric_limits<double>::max();}")
        # print("Third")
        # df = df.Define("max_dropped_seed_pt", "auto it = std::max_element(dropped_seed_pt.begin(), dropped_seed_pt.end()); if(it != dropped_seed_pt.end()){return *it;}else{return std::numeric_limits<double>::lowest();}")
        # print("Fourth")

        # sums = [df.Sum("n_dropped_seed_pix"), df.Sum("sum_dropped_seed_pt"),
        #         df.Min("min_dropped_seed_pt"), df.Max("max_dropped_seed_pt")]
        
        # print("Initial sums")
        # print(s, "rdf n seeds:", sums[0].GetValue())
        # print(s, "rdf pt sum:", sums[1].GetValue())
        # print(s, "rdf pt min:", sums[2].GetValue())
        # print(s, "rdf pt max:", sums[3].GetValue())
