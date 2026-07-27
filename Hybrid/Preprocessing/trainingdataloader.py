from __future__ import annotations

import awkward as ak
from fastml.utils.image import sliding_window
from fastml.utils_egamma.efex import eFex_slidingwindow_mask
import numpy as np
from sklearn.model_selection import train_test_split

from .dataloader import OpenDataLoader
from .datahelper import *
from .trainingdatahelper import *


class OpenDataSet():
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val   = None
        self.y_val   = None
        self.w_train = None
        self.w_val = None

        self.bin_edges = np.arange(10.0, 40.0 + 0.5, 0.5)
        self.thresholds = [(i, i+0.5) for i in np.arange(10, 40,0.5)]
    
    def load_raw_data(self):
        name_map = {
            "Zee": "zee",
            "JZ": "jz",
        }
        loaders = {}
        towers_dict = {}

        for sample_name, _ in name_map.items():
            dl = OpenDataLoader(sample_name)
            stop = 3 if 'Zee' in sample_name else 2

            dl.load(n_start=0, n_stop=stop)

            w = ak.broadcast_arrays(dl.weight, dl.seed_vectors.eta)[0]

            _, (ev_ids, e0, p0) = eFex_slidingwindow_mask(dl.seed_vectors)

            towers = sliding_window(dl.towers, 3)[ev_ids, e0, p0]

            dl.seed_vectors = ak.flatten(dl.seed_vectors)
            dl.weight = ak.to_numpy(ak.flatten(w))
            towers_dict[sample_name] = towers
            loaders[sample_name] = dl
        return loaders, towers_dict

    def load(self):
        loaders, towers_dict = self.load_raw_data()

        training_tower_dist, training_pt_dist = filter_training(loaders, towers_dict, self.thresholds)

        X_bkg = training_tower_dist['JZ'] 

        total_bkg = training_pt_dist['JZ']
        
        X_sig = training_tower_dist['Zee']
        
        y_bkg = np.zeros((len(X_bkg), 1, 1, 1), dtype=np.int8)
        y_sig = np.ones((len(X_sig), 1, 1, 1), dtype=np.int8)

        w_bkg = calculate_weights(total_bkg, self.bin_edges)
        w_sig = calculate_weights(training_pt_dist['Zee'], self.bin_edges)
        
        X = np.concatenate([X_sig, X_bkg], axis=0)
        y = np.concatenate([y_sig, y_bkg], axis=0)
        w = np.concatenate([w_sig, w_bkg], axis=0)

        X_cnn_train, X_cnn_test, y_train, y_test, w_train, w_test = train_test_split(
            X,
            y,
            w,
            test_size=0.2,
            random_state=101,
        )

        self.x_train = X_cnn_train
        self.y_train = y_train
        self.x_val   = X_cnn_test
        self.y_val   = y_test
        self.w_train = w_train 
        self.w_val   = w_test

    def save_to_root(self, train_path, val_path):
        def save_sample_to_root(sample, x, y, w, path):
            x_final = np.ascontiguousarray(
                x.reshape(len(x), -1),
                dtype=np.float64
            )

            y_matrix = y.reshape(len(y), -1)
            y_final = np.ascontiguousarray(
                y_matrix[:, 0], 
                dtype=np.int32
            )

            w_matrix = w.reshape(len(w), -1)
            w_final = np.ascontiguousarray(
                w_matrix[:, 0],
                dtype=np.float64
            )

            data = {
                    f"x_{sample}": x_final,
                    f"y_{sample}": y_final,
                    f"w_{sample}": w_final
            }
            df = ak.to_rdataframe(data)
            df.AsNumpy() # Perhaps required to avoid a segfault
            df.Snapshot("tree", path)

        save_sample_to_root(
            "train", 
            self.x_train, self.y_train, self.w_train, 
            train_path
        )
        save_sample_to_root(
            "val", 
            self.x_val, self.y_val, self.w_val, 
            val_path
        )

    

if __name__ == "__main__":
    ods = OpenDataSet()
    ods.load()
    ods.save_to_root(
        "Hybrid/Data/train.root", 
        "Hybrid/Data/val.root"
    )
