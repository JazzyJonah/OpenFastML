import ROOT
from fastml.utils.misc import get_config
import numpy as np

ROOT.DisableImplicitMT()

from opendataloader import OpenDataLoader
from rootdatahelper import get_root_data, add_towers



class RootDataLoader(OpenDataLoader):
    def load(self, n_start = None, n_stop = None):
        config = get_config(self.sample_name)
        for no_PU in [True, False]:
            self.df = get_root_data(
                config,
                n_start,
                n_stop,
                no_PU
            )

            self.df = add_towers(self.df, no_PU)
        self._load_seeddata


    def save_to_root(self, save_path):
        print(len(self.x_train))
        return
        train_df = ROOT.RDataFrame(len(self.x_train))
        train_data = {
            "x_train": self.x_train,
            "y_train": self.y_train,
            "w_train": self.w_train
        }



if __name__=="__main__":
    dl = RootDataLoader("Zee")
    dl.load(n_start=0, n_stop=2)
