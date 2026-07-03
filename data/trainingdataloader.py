import awkward as ak
import os
from fastml.utils.image import sliding_window
from fastml.utils_egamma.efex import eFex_slidingwindow_mask
import numpy as np
from sklearn.model_selection import train_test_split

from data.opendataloader import OpenDataLoader
from data.opendatahelper import *
from data.trainingdatahelper import *


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
            stop = 2 if 'Zee' in sample_name else 1

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

        print("JZ towers:", training_tower_dist["JZ"].shape)
        print("Zee towers:", training_tower_dist["Zee"].shape)

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
        print("X:", X.shape)
        print("y:", y.shape)
        print("w:", w.shape)

        X_cnn_train, X_cnn_test, y_train, y_test, w_train, w_test = train_test_split(
            X,
            y,
            w,
            test_size=0.2,
            random_state=101,
        )
        print("x_train:", X_cnn_train.shape)

        self.x_train = X_cnn_train
        self.y_train = y_train
        self.x_val   = X_cnn_test
        self.y_val   = y_test
        self.w_train = w_train 
        self.w_val   = w_test

    def save_to_parquet(self, save_path):
        train_data = {
        "x_train": self.x_train,
        "y_train": self.y_train,
        "w_train": self.w_train
        }

        train_data = ak.zip(train_data, depth_limit=1)
        ak.to_parquet(train_data, save_path)
        
        val_data = {
        "x_val": self.x_val,
        "y_val": self.y_val,
        "w_val": self.w_val,
        }

        val_data = ak.zip(val_data, depth_limit=1)
        ak.to_parquet(val_data, save_path)

if __name__ == "__main__":
    ods = OpenDataSet()
    ods.load()