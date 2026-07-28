from Root_single.Preprocessing.datahelper import (get_config,
                        get_root_data, 
                        add_towers, 
                        add_seed_and_truth_vectors)

class RootDataLoader:
    def __init__(self, sample_name):
        self.sample_name = sample_name

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