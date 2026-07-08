import ROOT
from fastml.utils.misc import get_config
import numpy as np

from opendataloader import OpenDataLoader
from rootdatahelper import (get_root_data, 
                            add_towers, 
                            add_seed_and_truth_vectors)

class RootDataLoader(OpenDataLoader):
    def load(self, n_start = None, n_stop = None):
        config = get_config(self.sample_name)

        self.df, self._temp = get_root_data(
            config,
            n_start,
            n_stop,
        )
        self.df = add_towers(self.df)

        self._load_seeddata()

    def _load_seeddata(self):
        self.df = add_seed_and_truth_vectors(self.df, self.sample_name)
        

if __name__=="__main__":
    dl = RootDataLoader("Zee")
    dl.load(n_start=0, n_stop=2)
    dl = RootDataLoader("JZ")
    dl.load(n_start=0, n_stop=1)
