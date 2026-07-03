from trainingdataloader import OpenDataSet
from rootdataloader import RootDataLoader
from rootdatahelper import _make_get_bin_count

import numpy as np

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
        print("HERE")

        def _sum_vec(x):
            return np.float64(np.sum(x))

        def _min_vec(x):
            return np.float64(np.min(x)) if len(x) else np.inf

        def _max_vec(x):
            return np.float64(np.max(x)) if len(x) else -np.inf

        # for s in ["Zee", "JZ"]:
        s = "JZ"
        df = self.loaders[s].df

        df = df.Define("n_dropped_seed_pix", "dropped_seed_pix.size()")
        df = df.Define("sum_dropped_seed_pt", "std::accumulate(dropped_seed_pt.begin(), dropped_seed_pt.end(), 0.)")
        df = df.Define("min_dropped_seed_pt", "auto it = std::min_element(dropped_seed_pt.begin(), dropped_seed_pt.end()); if(it != dropped_seed_pt.end()){return *it;}else{return std::numeric_limits<double>::max();}")
        df = df.Define("max_dropped_seed_pt", "auto it = std::max_element(dropped_seed_pt.begin(), dropped_seed_pt.end()); if(it != dropped_seed_pt.end()){return *it;}else{return std::numeric_limits<double>::lowest();}")

        sums = [df.Sum("n_dropped_seed_pix"), df.Sum("sum_dropped_seed_pt"),
                df.Min("min_dropped_seed_pt"), df.Max("max_dropped_seed_pt")]
        
        print(s, "rdf n seeds:", sums[0].GetValue())
        print(s, "rdf pt sum:", sums[1].GetValue())
        print(s, "rdf pt min:", sums[2].GetValue())
        print(s, "rdf pt max:", sums[3].GetValue())
            
        # self.compute_training_targets('JZ')
        # self.test('JZ')

    def test(self, sample_name):
        df_dbg = self.loaders["JZ"].df.Range(10)

        for i in [0, 1, 2, 3, 10, 50, 99]:
            df_dbg = df_dbg.Define(
                f"dbg_bin_{i}",
                f"static_cast<Long64_t>(seed_bin_counts[{i}])"
            )

        df_dbg.Display(
            [
                "seed_bin_counts",
                "dbg_bin_0",
                "dbg_bin_1",
                "dbg_bin_2",
                "dbg_bin_3",
                "dbg_bin_10",
                "dbg_bin_50",
                "dbg_bin_99",
            ],
            10
        ).Print()
    

    def compute_training_targets(self, sample_name):
        # print(self._sum_seed_bin_contents)
        df_test = self.loaders[sample_name].df.Range(100)

        rdf_totals = []
        df_tmp = df_test

        for i in range(len(self.thresholds)):
            col = f"seed_bin_count_{i}"
            df_tmp = df_tmp.Define(
                col, 
                f"seed_bin_counts[{i}]",
                #_make_get_bin_count(i), ["seed_bin_counts"]
            )   
            rdf_totals.append(df_tmp.Sum(col))

        rdf_totals = np.asarray([h.GetValue() for h in rdf_totals], dtype=np.int64)

        arr = df_test.AsNumpy(["seed_bin_counts"])["seed_bin_counts"]
        np_totals = np.stack([np.asarray(x, dtype=np.int64) for x in arr]).sum(axis=0)

        print(rdf_totals)
        print(np_totals)
        print(np.array_equal(rdf_totals, np_totals))

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