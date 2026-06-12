import awkward as ak
import vector
import numpy as np
import tensorflow as tf

metre = 1e3


def remove_transition(cells):
    cell_abseta = np.abs(to_3vector(cells).eta)
    psb = (cells.cell_sampling == 0) & (cell_abseta > 1.5)
    eme1 = (cells.cell_sampling == 5) & (cell_abseta < 1.5)
    eme2 = (cells.cell_sampling == 6) & (cell_abseta < 1.5)
    ext0 = (cells.cell_sampling == 18) & (cell_abseta > 1.5)
    mask = (~psb) & (~eme1) & (~eme2) & (~ext0)
    return cells[mask]


def get_layer(sampling):
    layer_map = {
        0: 0,  # PSB
        1: 1,  # EMB1
        2: 2,  # EMB2
        3: 3,  # EMB3
        4: 0,  # PSE
        5: 1,  # EME1
        6: 2,  # EME2
        7: 3,  # EME3
        8: 4,  # HEC0
        9: 5,  # HEC1
        10: 6,  # HEC2
        11: 7,  # HEC3
        12: 4,  # TileBar0
        13: 5,  # TileBar1
        14: 6,  # TileBar2
        15: 5,  # TileGap1
        16: 6,  # TileGap2
        17: 7,  # TileGap3
        18: 4,  # TileExt0
        19: 5,  # TileExt1
        20: 6,  # TileExt2
        # things get weird in the fcal
        21: 1,  # FCAL0 (EM)
        22: 2,  # FCAL1 (Had)
        23: 3,  # FCAL1 (Had)
    }

    output = ak.copy(sampling)
    for k, v in layer_map.items():
        output = ak.where(output == k, v, output)

    return output


def to_3vector(cells):
    vectors = vector.zip(
        {
            "x": cells.cell_x / metre,
            "y": cells.cell_y / metre,
            "z": cells.cell_z / metre,
        }
    )
    return vectors


def to_4momentum(cells, Et_key="cell_et"):
    position = to_3vector(cells)
    vectors = vector.zip(
        {
            "m": ak.zeros_like(cells[Et_key]),
            "pt": cells[Et_key],
            "eta": position.eta,
            "phi": position.phi,
        }
    )
    return vectors


def unflatten(x, like):
    out = []
    start_idx = 0
    for length in ak.count(like, axis=1):
        if length > 0:
            out.append(x[start_idx : start_idx + length])

        else:
            out.append([])

        start_idx += length

    return ak.Array(out)


def encode(x):
    if isinstance(x, np.ndarray):
        where = np.where
    elif isinstance(x, ak.Array):
        where = ak.where
    elif isinstance(x, tf.Tensor):
        where = tf.where
    else:
        raise ValueError("unrecognised input type.")

    linear_1 = where((0 <= x) & (x < 8), x // 0.03125, 0)
    linear_2 = where((8 <= x) & (x < 40), 256 + (x - 8) // 0.125, 0)
    linear_3 = where((40 <= x) & (x < 168), 512 + (x - 40) // 0.5, 0)
    linear_4 = where((168 <= x) & (x < 678), 768 + (x - 168) // 2.0, 0)
    overflow = where(678 <= x, 1023, 0)

    return linear_1 + linear_2 + linear_3 + linear_4 + overflow


def decode(x):
    if isinstance(x, np.ndarray):
        where = np.where
    elif isinstance(x, ak.Array):
        where = ak.where
    elif isinstance(x, tf.Tensor):
        where = tf.where
    else:
        raise ValueError("unrecognised input type.")

    linear_1 = where((0 <= x) & (x < 256), x * 0.03125, 0)
    linear_2 = where((256 <= x) & (x < 512), 8 + (x - 256) * 0.125, 0)
    linear_3 = where((512 <= x) & (x < 768), 40 + (x - 512) * 0.5, 0)
    linear_4 = where((768 <= x) & (x < 1023), 168 + (x - 768) * 2.0, 0)
    overflow = where(1023 <= x, 678, 0)

    return linear_1 + linear_2 + linear_3 + linear_4 + overflow


def fixed_encoding(x):
    return encode(x) / 2**5


def fixed_decoding(x):
    return decode(x) * 2**5


def quantise(x):
    return decode(encode(x))
