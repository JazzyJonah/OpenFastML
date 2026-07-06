from trainingdataloader import OpenDataSet
from rootdataloader import RootDataLoader

import numpy as np
import awkward as ak

class RootDataSet(OpenDataSet):
    def load_raw_data(self) -> None:
        samples = ["Zee", "JZ"]
        self.loaders: dict[str, RootDataLoader] = {}

        for sample_name in samples:
            dl = RootDataLoader(sample_name)
            stop = 2 if 'Zee' == sample_name else 1

            dl.load(self.thresholds, n_start=0, n_stop=stop)

            self.loaders[sample_name] = dl
    

    def load(self):
        self.load_raw_data()
        print("Loaded raw data")

        self.convert_loaders_to_awkward()
        print("Converted loaders to awkward")

        self.rechunk_seed_x_by_sample()
        print("Rechunked seed_x by sample")

        self.group_samples_into_threshold_bins()
        print("Grouped samples into threshold bins")

        self.build_binned_table()
        print("Built binned table")

        self.make_binned_df()
        print("Made binned RDataFrame")


    def convert_loaders_to_awkward(self):
        self.arrays_by_sample = {
            sample_name: self._convert_loader_to_awkward(sample_name) 
            for sample_name in ("Zee", "JZ")}
    
    def _convert_loader_to_awkward(self, sample_name):
        loader = self.loaders[sample_name]
        columns = (
            "dropped_seed_pix",
            "dropped_seed_pt",
            "seed_x_bank"
        )
        return ak.from_rdataframe(
            loader.df,
            columns=columns,
            keep_order=True
        )
    
    
    def rechunk_seed_x_by_sample(self):
        self.arrays_by_sample = {
            sample: self._rechunk_seed_x(arr)
            for sample, arr in self.arrays_by_sample.items()}
    
    def _rechunk_seed_x(self, arr):
        """
        Add arr.seed_x with structure:

            event -> seed -> 18 values

        starting from arr.seed_x_bank with structure:

            event -> flat vector of length 18 * n_seeds
        """

        seed_x = ak.unflatten(arr.seed_x_bank, 3 * 3 * 2, axis=1)
        seed_x = ak.to_regular(seed_x, axis=2)

        return ak.with_field(arr, seed_x, "seed_x")
    

    def group_samples_into_threshold_bins(self):
        self.binned_by_sample = {
            sample: self._group_sample_into_threshold_bins(arr)
            for sample, arr in self.arrays_by_sample.items()}

    def _group_sample_into_threshold_bins(self, arr):
        """
        Input arr fields:
            dropped_seed_pix: event -> seed
            dropped_seed_pt:  event -> seed
            seed_x:           event -> seed -> 18

        Output:
            binned: threshold_bin -> {
                seeds: variable number of 18-vectors,
                pt:    variable number of floats,
                pix:   variable number of ints
            }

        Outer length = len(self.thresholds), e.g. 60.
        """

        n_bins = len(self.thresholds)

        # Flatten only the event dimension.
        # Keep seed_x as seed -> 18.
        pt_flat = ak.to_numpy(ak.flatten(arr.dropped_seed_pt, axis=1))
        pix_flat = ak.to_numpy(ak.flatten(arr.dropped_seed_pix, axis=1))
        x_flat = ak.flatten(arr.seed_x, axis=1)

        # Build threshold edges from [(lo0, hi0), (lo1, hi1), ...]
        edges = np.asarray(
            [self.thresholds[0][0]] + [hi for _, hi in self.thresholds],
            dtype=np.float64,
        )

        # Same convention as original:
        # normal bins: [lo, hi)
        # final bin:  [lo, hi]
        bin_id = np.searchsorted(edges, pt_flat, side="right") - 1

        # Include exact final upper edge in final bin.
        bin_id[pt_flat == edges[-1]] = n_bins - 1

        valid = (bin_id >= 0) & (bin_id < n_bins)

        pt_valid = pt_flat[valid]
        pix_valid = pix_flat[valid]
        x_valid = x_flat[valid]
        bin_valid = bin_id[valid]

        # Group by threshold bin.
        order = np.argsort(bin_valid, kind="stable")

        pt_sorted = pt_valid[order]
        pix_sorted = pix_valid[order]
        x_sorted = x_valid[order]
        bin_sorted = bin_valid[order]

        counts = np.bincount(bin_sorted, minlength=n_bins)

        binned_pt = ak.unflatten(pt_sorted, counts)
        binned_pix = ak.unflatten(pix_sorted, counts)
        binned_seeds = ak.unflatten(x_sorted, counts)

        return ak.zip(
            {
                "seeds": binned_seeds,
                "pt": binned_pt,
                "pix": binned_pix,
            },
            depth_limit=1,
        )
    

    def build_binned_table(self):
        sig = self.binned_by_sample["Zee"]
        bkg = self.binned_by_sample["JZ"]

        # Each `*.seeds` is currently:
        #     threshold_bin -> seed -> 18
        #
        # Flatten only the seed->18 level, so each threshold bin has:
        #     threshold_bin -> flat length 18 * n_seeds
        signal_seeds = ak.flatten(sig.seeds, axis=2)
        background_seeds = ak.flatten(bkg.seeds, axis=2)

        self.binned_table = ak.Array({
            "signal_seeds": signal_seeds,
            "background_seeds": background_seeds,

            "signal_pt": sig.pt,
            "background_pt": bkg.pt,

            # optional, but very useful for debugging
            "signal_pix": sig.pix,
            "background_pix": bkg.pix,
        })


    def make_binned_df(self):
        if not hasattr(self, "binned_table"):
            self.build_binned_table()

        self.binned_df = ak.to_rdataframe({
            "signal_seeds": self.binned_table.signal_seeds,
            "background_seeds": self.binned_table.background_seeds,
            "signal_pt": self.binned_table.signal_pt,
            "background_pt": self.binned_table.background_pt,
            "signal_pix": self.binned_table.signal_pix,
            "background_pix": self.binned_table.background_pix,
        })        


    def _sum_seed_bin_contents(self, sample_name):
        df = self.loaders[sample_name].df
        totals = []
        for i in range(len(self.thresholds)):
            col = f"seed_bin_count_{i}"
            df_i = df.Define(
                col,
                f"seed_bin_counts[{i}]"
            )
            totals.append(df_i.Sum(col))

        return np.asarray(
            [total.GetValue() for total in totals], 
            dtype=np.int64
        )
    

if __name__ == "__main__":
    rds = RootDataSet()
    rds.load()