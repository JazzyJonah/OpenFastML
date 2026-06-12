import awkward as ak
import vector
import numpy as np
import json
import os
import uproot
import functools

print = functools.partial(print, flush=True)


def get_config(sample):
    return json.load(open("data/configs/samples.json"))[sample]


def get_data(config, **kwargs):
    files = [
        config["dir"] + "/" + f
        for f in os.listdir(config["dir"])
        if config["keyword"] in f and ".root" in f
    ]
    events = uproot.concatenate(files, **kwargs)

    if "EventWeight" in events.fields:
        weights = ak.to_numpy(events["EventWeight"][:, 0])
        events["weight"] = (
            weights
            * config["xsec"]
            * config["hstp_filter_sf"]
            * config["filter_eff"]
            / weights.sum()
        )
        events = ak.without_field(events, where="EventWeight")

    for var in ["cell_et", "cell_et_mu0"]:
        if var in events.fields:
            events[var] = events[var] / 1000  # MeV --> GeV

    return events


def sparse_to_awkward(arr):
    mask = arr.pt > 0
    counts = ak.from_numpy(np.sum(mask.reshape(mask.shape[0], -1), axis=1))
    m = ak.unflatten(arr.m[mask], counts)
    pt = ak.unflatten(arr.pt[mask], counts)
    eta = ak.unflatten(arr.eta[mask], counts)
    phi = ak.unflatten(arr.phi[mask], counts)
    return vector.zip({"m": m, "pt": pt, "eta": eta, "phi": phi})


def sort_and_pad(vectors, n=6):
    vectors = vectors[ak.argsort(vectors.pt, axis=1, ascending=False)]

    padded_vectors = vector.zip(
        {
            "m": ak.fill_none(ak.pad_none(vectors.m, n, clip=True), 0),
            "pt": ak.fill_none(ak.pad_none(vectors.pt, n, clip=True), 0),
            "eta": ak.fill_none(ak.pad_none(vectors.eta, n, clip=True), 0),
            "phi": ak.fill_none(ak.pad_none(vectors.phi, n, clip=True), 0),
        }
    )

    return padded_vectors


def awkward_to_vector(obj):
    if "pt" in obj.fields:
        return vector.zip({"pt": obj.pt, "eta": obj.eta, "phi": obj.phi, "m": obj.m})

    if "rho" in obj.fields:
        return vector.zip({"pt": obj.rho, "eta": obj.eta, "phi": obj.phi, "m": obj.tau})

    elif "t" in obj.fields:
        return vector.zip({"px": obj.x, "py": obj.y, "pz": obj.z, "E": obj.t})

    elif "px" in obj.fields:
        return vector.zip({"px": obj.px, "py": obj.py, "pz": obj.pz, "E": obj.E})

    else:
        raise ValueError("could not detect vector component keys.")


def balance_weights(weights, labels):
    classes = np.unique(labels)
    for y in classes:
        weights[np.nonzero(labels == y)] *= (
            len(labels) / len(classes) / np.sum(weights[labels == y])
        )
    return weights


def balance_indices(idxs0, idxs1):
    if len(idxs1) > len(idxs0):
        idxs1 = np.random.choice(idxs1, len(idxs0), replace=False)

    elif len(idxs1) < len(idxs0):
        idxs0 = np.random.choice(idxs0, len(idxs1), replace=False)

    return np.concatenate((idxs1, idxs0))
