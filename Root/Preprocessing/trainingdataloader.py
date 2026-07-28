from Root.Preprocessing.dataloader import RootDataLoader
from Root.Preprocessing.trainingdatahelper import (_group_sample_into_threshold_bins,
                                    _convert_loader_to_awkward,
                                    _rechunk_seed_x,
                                    add_signal_background_counts,
                                    add_smooth_raw_targets,
                                    add_final_seeds,
                                    calculate_weights)

import awkward as ak
import numpy as np
from sklearn.model_selection import train_test_split

class RootDataSet:
    def __init__(self):
        self.bin_edges = np.arange(10.0, 40.0 + 0.5, 0.5)
        self.thresholds = [(i, i+0.5) for i in np.arange(10, 40,0.5)]

    def load_raw_data(self) -> None:
        samples = ["Zee", "JZ"]
        self.loaders: dict[str, RootDataLoader] = {}

        for sample_name in samples:
            dl = RootDataLoader(sample_name)
            stop = 3 if 'Zee' == sample_name else 2

            dl.load(n_start=0, n_stop=stop)

            self.loaders[sample_name] = dl

    def load(self) -> None:
        self.load_raw_data()
        self.arrays_by_sample = self.convert_loaders_to_awkward()
        self.arrays_by_sample = self.rechunk_seed_x_by_sample()
        self.binned_by_sample = self.group_samples_into_threshold_bins()
        self.binned_table = self.build_binned_table()
        self.binned_df = self.make_binned_df()
        self.binned_df = add_signal_background_counts(self.binned_df)
        self.binned_df = add_smooth_raw_targets(self.binned_df)
        self.binned_df = add_final_seeds(self.binned_df)
        self.selected_binned_array = self.selected_binned_df_to_awkward()
        self.final_seed_table = self.make_final_seed_table()
        self.final_df = self.make_final_df()    

    def save_to_root(self, train_path, val_path) -> None:
        self.final_df[0].Display().Print()
        self.final_df[0].Snapshot("tree", train_path)
        self.final_df[1].Display().Print()
        self.final_df[1].Snapshot("tree", val_path)
    
    def convert_loaders_to_awkward(self) -> dict[str, ak.Array]:
        """Convert loaders to a dictionary of Awkward arrays and return"""
        return {
            sample_name: _convert_loader_to_awkward(self.loaders[sample_name].df) 
            for sample_name in ("Zee", "JZ")}
    
    def rechunk_seed_x_by_sample(self) -> dict[str, ak.Array]:
        """Add "seed_x" field, chunked in 18-length segments, \
            to self.arrays_by_sample[s] and return"""
        return {
            sample: _rechunk_seed_x(arr)
            for sample, arr in self.arrays_by_sample.items()}

    def group_samples_into_threshold_bins(self) -> dict[str, ak.Array]:
        """Create and return dict of len(self.thresholds), e.g. 60,-length ]
            Awkward array with the following fields: "seeds", "pt", "pix"."""
        return {
            sample: _group_sample_into_threshold_bins(arr, self.thresholds)
            for sample, arr in self.arrays_by_sample.items()}

    def build_binned_table(self) -> ak.Array:
        """Create and return len(self.thresholds), e.g. 60,-length\
            Awkward array with all fields necessary for second RDF
            
            return fields: "signal_seeds", "background_seeds", \
                "signal_pt", "background_pt", signal_pix", "background_pix"."""
        sig = self.binned_by_sample["Zee"]
        bkg = self.binned_by_sample["JZ"]

        signal_seeds = ak.flatten(sig.seeds, axis=2)
        background_seeds = ak.flatten(bkg.seeds, axis=2)

        return ak.Array({
            "signal_seeds": signal_seeds,
            "background_seeds": background_seeds,

            "signal_pt": sig.pt,
            "background_pt": bkg.pt,

            "signal_pix": sig.pix,
            "background_pix": bkg.pix,
        })

    def make_binned_df(self):
        """Return an RDataFrame with the following columns: \
        "signal_seeds", "background_seeds", "signal_pt", \
            "background_pt", signal_pix", "background_pix"."""
        
        return ak.to_rdataframe({
            "signal_seeds": self.binned_table.signal_seeds,
            "background_seeds": self.binned_table.background_seeds,
            "signal_pt": self.binned_table.signal_pt,
            "background_pt": self.binned_table.background_pt,
            "signal_pix": self.binned_table.signal_pix,
            "background_pix": self.binned_table.background_pix,
        })

    def selected_binned_df_to_awkward(self) -> ak.Array:
        """Return an Awkward array with the following fields: \
        "final_signal_seeds", "final_background_seeds", \
            "final_signal_pt", "final_background_pt"."""
        
        return ak.from_rdataframe(
            self.binned_df,
            columns=[
                "final_signal_seeds",
                "final_background_seeds",
                "final_signal_pt",
                "final_background_pt"
            ],
            keep_order=True
        )
    
    def make_final_seed_table(self) -> ak.Array:
        """Return final seed table, ready to export back to \
            RDF for snapshotting. Fields: "x", "weight", "label"."""
        
        arr=self.selected_binned_array

        sig_x = ak.flatten(
            ak.unflatten(arr.final_signal_seeds, 18, axis=1),
            axis=1)
        bkg_x = ak.flatten(
            ak.unflatten(arr.final_background_seeds, 18, axis=1),
            axis=1)

        sig_pt = ak.flatten(arr.final_signal_pt, axis=1)
        bkg_pt = ak.flatten(arr.final_background_pt, axis=1)
        sig_w = calculate_weights(sig_pt, self.bin_edges)
        bkg_w = calculate_weights(bkg_pt, self.bin_edges)

        sig_table = ak.Array({
            "x": sig_x,
            "y": np.ones(len(sig_pt)),
            "w": sig_w
        })
        bkg_table = ak.Array({
            "x": bkg_x,
            "y": np.zeros(len(bkg_pt)),
            "w": bkg_w
        })

        return ak.concatenate([sig_table, bkg_table], axis=0)
    
    def make_final_df(self):
        x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
            self.final_seed_table.x,
            self.final_seed_table.y,
            self.final_seed_table.w,
            test_size=0.2,
            random_state=101
        )
        df_train = ak.to_rdataframe({
            "x_train": x_train,
            "y_train": y_train,
            "w_train": w_train
        })
        df_val = ak.to_rdataframe({
            "x_val": x_val,
            "y_val": y_val,
            "w_val": w_val
        })
        return df_train, df_val


if __name__ == "__main__":
    rds = RootDataSet()
    rds.load()
    rds.save_to_root(
        "Root/Data/train.root",
        "Root/Data/val.root"
    )