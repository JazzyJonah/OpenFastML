import awkward as ak
import numpy as np
import vector

from fastml.utils.cells import to_3vector, to_4momentum


def to_topoclusters(cells):
    topo_cells = cells[cells.cell_topo422 != -1]
    topo_cells = topo_cells[
        ak.argsort(topo_cells.cell_topo422, axis=1, ascending=False)
    ]

    n_topo = ak.run_lengths(topo_cells.cell_topo422)
    counts = ak.flatten(n_topo, axis=None)

    topoclusters = {}
    for k in topo_cells.fields:
        topoclusters[k] = ak.unflatten(topo_cells[k], counts, axis=-1)

    return ak.zip(topoclusters)


def cluster_vecsum(topo):
    return ak.sum(to_4momentum(topo), axis=-1)


def to_topo_graph(topoclusters, max_adj, target="zvtx"):
    topo = topoclusters[ak.argsort(topoclusters.cell_et, axis=-1, ascending=False)]

    topo_p4 = to_4momentum(topo)
    topo_x3 = to_3vector(topo)

    topo_p4_sum = cluster_vecsum(topo)

    graph_features = {
        "logEt": np.log(1 + topo_p4.pt),
        "r": topo_x3.mag,
        "eta": topo_p4.eta,
        "dphi": topo_p4.deltaphi(topo_p4_sum),
        target: topo[target],
    }

    for k, v in graph_features.items():
        graph_features[k] = ak.fill_none(ak.pad_none(v, max_adj, axis=2, clip=True), 0)

    return ak.zip(graph_features)


def predict(model, topoclusters, max_adj):
    graphs = ak.flatten(to_topo_graph(topoclusters, max_adj=max_adj))
    X = np.stack([ak.to_numpy(graphs[k]) for k in ["Et", "r", "eta", "dphi"]], axis=-1)
    pred = model.predict(X, batch_size=1024, verbose=2)
    topo_pred = ak.unflatten(pred.flatten(), ak.num(topoclusters, axis=1))
    return topo_pred


def recluster(model, topoclusters, max_adj):
    pred_Et = predict(model, topoclusters, max_adj)
    clusters = cluster_vecsum(topoclusters)
    new_clusters = vector.zip(
        {"m": clusters.m, "pt": pred_Et, "eta": clusters.eta, "phi": clusters.phi}
    )
    return new_clusters
