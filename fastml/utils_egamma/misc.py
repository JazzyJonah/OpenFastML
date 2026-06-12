import os
import uproot
import awkward as ak
import numpy as np

def get_open_data(config, n_start=None, n_stop=None, **kwargs):
    """
    "Get data but adapted to open data"
    "n_start, n_stop": counts of 10k files loaded
    """
    path = config["dir"]
    no_PU = kwargs.get("no_PU", False)

    branches = [
        'Event/Event.Weight',
        'Event/Event.CrossSection',
        'Particle/Particle.PID',
        'Tower/Tower.Eem',
        'Tower/Tower.Ehad',
        'Tower/Tower.Eta',
        'Tower/Tower.Phi',
    ]

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
        
    events = uproot.concatenate(
        ({f: "Delphes"} for f in files),
        branches,
        library="ak"
    )

    if "Event/Event.Weight" in events.fields:
        weights = ak.to_numpy(events["Event/Event.Weight"][:, 0])
        events['Event/Event.Weight'] = (
            weights
            * 1 # xsec
            * 1 # filter eff
            / weights.sum()
        )
        
    return events

def get_dR_matrices(vectors, truth):
    """
    vectors: ak.Array of per-event eFEX vectors (jagged)
    truth:   per-event truth object (either a single 4-vector or None)
             e.g. your lead_truth = ak.firsts(items.truth_vectors[i_truth])
    return:  ak.Array of same "outer" structure as `vectors`,
             each entry is the min ΔR to the truth object, or +inf if no truth.
             New: ak.Array
    """

    truth_as_list = ak.singletons(truth)  
    truth_masked = ak.mask(truth_as_list, ~ak.is_none(truth_as_list))

    pairs = ak.cartesian([vectors, truth_masked], axis=1, nested=True)
    v = pairs["0"]
    t = pairs["1"] 

    dR = v.deltaR(t)
    dR = ak.fill_none(dR, np.inf)
    min_dR = ak.min(dR, axis=-1)
    min_dR = ak.fill_none(min_dR, np.inf)

    truth_ix = ak.argmin(dR, axis=-1) 
    truth_ix = ak.fill_none(truth_ix, -1)
    truth_ix = ak.firsts(truth_ix, axis=-1)
    valid = truth_ix >= 0
    truth_ix = truth_ix[valid]

    return min_dR, truth_ix