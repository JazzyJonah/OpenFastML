import numpy as np
import skimage
import tensorflow as tf
import vector
import awkward as ak

from fastml.utils.cells import get_layer, to_4momentum, remove_transition


def vector_to_tower(
    vectors,
    eta_edges=np.linspace(-2.5, 2.5, 51),
    phi_edges=np.linspace(-np.pi, np.pi, 65),
):
    tower_edges = (np.arange(1 + len(vectors)), eta_edges, phi_edges)

    event_indices = ak.flatten(get_index(vectors))
    flat_vectors = ak.flatten(vectors)

    towers = np.histogramdd(
        (
            ak.to_numpy(event_indices),
            ak.to_numpy(flat_vectors.eta),
            ak.to_numpy(flat_vectors.phi),
        ),
        bins=tower_edges,
        weights=ak.to_numpy(flat_vectors.pt),
    )[0]

    return np.expand_dims(towers, axis=-1)


def cells_to_towers(cells, Et_key="cell_et"):
    cells = remove_transition(cells)
    cell_layer = get_layer(cells.cell_sampling)
    cell_vectors = to_4momentum(cells, Et_key=Et_key)

    cell_eta = cell_vectors.eta
    cell_eta = (
        cell_eta
        + ak.where((cell_eta > 1.5) & (cell_layer == 2), -0.01, 0)
        + ak.where((cell_eta < -1.5) & (cell_layer == 2), +0.01, 0)
        + ak.where((cell_eta > 0.1) & (cell_eta < 1.4) & (cell_layer == 1), -0.005, 0)
    )

    cell_vectors = vector.zip(
        {
            "pt": cell_vectors.pt,
            "m": cell_vectors.m,
            "eta": cell_eta,
            "phi": cell_vectors.phi,
        }
    )

    towers = np.concatenate(
        [vector_to_tower(cell_vectors[cell_layer == layer]) for layer in range(6)],
        axis=-1,
    )

    return towers


def pad(x, pad_size):
    assert x.ndim == 4
    y = np.pad(
        x,
        ((0, 0), (pad_size, pad_size), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    y = np.pad(y, ((0, 0), (0, 0), (pad_size, pad_size), (0, 0)), mode="wrap")
    return y


def sliding_window(x, size):
    assert x.ndim == 4
    px = pad(x, (size - 1) // 2)
    windows = skimage.util.view_as_windows(
        px, window_shape=(1, size, size, px.shape[-1]), step=1
    )
    return windows[:,:,:,0,0,:,:,:]


def augment_image(X):
    aug_X = tf.concat(
        [
            X,
            tf.reverse(X, axis=[1]),
            tf.reverse(X, axis=[2]),
            tf.reverse(X, axis=[1, 2]),
        ],
        axis=0,
    )

    return aug_X


def augment_batch(X, y, w):
    X_aug = augment_image(X)

    y_multiples = [4] + [1] * (len(y.shape) - 1)
    w_multiples = [4] + [1] * (len(w.shape) - 1)

    y_aug = tf.tile(y, y_multiples)
    w_aug = tf.tile(w, w_multiples)

    return X_aug, y_aug, w_aug


def get_tower_eta(X):
    eta_idxs = np.indices(X[..., :1].shape)[1]
    return (eta_idxs - np.median(eta_idxs)) * 0.1


def tower_to_vector(X):
    _, eta_idxs, phi_idxs, _ = np.indices(X.shape)

    eta = (eta_idxs - np.median(eta_idxs)) * 0.1
    phi = (phi_idxs - np.median(phi_idxs)) * np.pi / 32

    vectors = vector.arr(
        {
            "eta": eta,
            "phi": phi,
            "pt": X,
            "m": np.zeros_like(X),
        }
    )

    return vectors


def get_index(vectors):
    return ak.ones_like(vectors.eta, dtype=int) * np.arange(len(vectors))
