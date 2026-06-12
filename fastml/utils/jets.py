import fastjet
import awkward as ak
import vector
import numpy as np
# import xgboost as xgb
import json
from sklearn.model_selection import train_test_split


def antikt_jets(vectors, min_pt, r=0.4):
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, r)
    cluster = fastjet.ClusterSequence(vectors, jetdef)
    return cluster.inclusive_jets(min_pt)


def truth_match(recon, truth, max_dR):
    if len(truth) != len(recon):
        raise ValueError("truth and recon jets must be same size along axis=0.")

    left, right = ak.unzip(ak.argcartesian([recon, truth]))

    dR = recon[left].deltaR(truth[right])

    is_close = dR < max_dR
    matched_truth = truth[right][is_close]
    matched_recon = recon[left][is_close]

    return matched_recon, matched_truth


class BinnedCalibration:
    def __init__(self, pT_bins=None, abseta_bins=None, max_dR=0.3):
        if pT_bins is None:
            pT_bins = np.array(
                [
                    5,
                    7.5,
                    10,
                    15,
                    20,
                    30,
                    40,
                    50,
                    60,
                    80,
                    90,
                    100,
                    150,
                    250,
                    500,
                    1000,
                ]
            )

        if abseta_bins is None:
            abseta_bins = np.array([0, 0.8, 1.5, 2, 2.5])

        self.bins = (pT_bins, abseta_bins)
        self.max_dR = max_dR

    def fit(self, truth_jets, recon_jets):
        matched_recon_jets, matched_truth_jets = truth_match(
            recon_jets, truth_jets, max_dR=self.max_dR
        )

        truth_pT = ak.to_numpy(ak.flatten(matched_truth_jets.pt))
        recon_pT = ak.to_numpy(ak.flatten(matched_recon_jets.pt))
        recon_abseta = np.abs(ak.to_numpy(ak.flatten(matched_recon_jets.eta)))

        k = np.histogram2d(
            recon_pT,
            recon_abseta,
            bins=self.bins,
            weights=recon_pT / truth_pT,
        )[0]

        N = np.histogram2d(
            recon_pT,
            recon_abseta,
            bins=self.bins,
        )[0]

        # num_empty_bins = np.sum(N ==0)
        # if num_empty_bins>0:
        #    raise ValueError(f"Found {num_empty_bins} empty bins! Increase bin size or use larger sample.")

        num_sparse_bins = np.sum(N < 10)
        if num_sparse_bins > 0:
            print(
                f"WARNING: Found {num_sparse_bins} / {N.size} bins with <10 matched jets"
            )

        self.scale_factors = np.divide(N, k, out=np.ones_like(N), where=k != 0)

        return self

    def predict(self, recon_jets):
        def get_bin_index(arr, bins):
            bin_idxs = np.digitize(ak.flatten(arr), bins) - 1
            return np.clip(bin_idxs, 0, len(bins) - 2)

        pt = recon_jets.pt
        abseta = np.abs(recon_jets.eta)

        pt_bins, abseta_bins = self.bins

        if ak.any(pt < np.min(pt_bins)):
            raise ValueError("found jet below lowest pT bin")

        if ak.any(abseta < np.min(abseta_bins)) or ak.any(abseta > np.max(abseta_bins)):
            raise ValueError("found jet outside abseta bin range")

        pt_bin_idxs = get_bin_index(pt, pt_bins)
        abseta_bin_idxs = get_bin_index(abseta, abseta_bins)

        scale_factors = self.scale_factors[pt_bin_idxs, abseta_bin_idxs]

        scale_factors = ak.unflatten(scale_factors, ak.num(recon_jets))

        scaled_recon_jets = {
            "m": recon_jets.m,
            "pt": recon_jets.pt * scale_factors,
            "eta": recon_jets.eta,
            "phi": recon_jets.phi,
        }

        return vector.zip(scaled_recon_jets)

    def save(self, path):
        pT_bins, abseta_bins = self.bins
        save_dict = {
            "pT_bins": pT_bins.tolist(),
            "abseta_bins": abseta_bins.tolist(),
            "scale_factors": self.scale_factors.tolist(),
            "max_dR": self.max_dR,
        }
        with open(path, "w") as f:
            json.dump(save_dict, f)

    def load(self, path):
        with open(path, "r") as f:
            load_dict = json.load(f)
        self.bins = (np.array(load_dict["pT_bins"]), np.array(load_dict["abseta_bins"]))
        self.scale_factors = np.array(load_dict["scale_factors"])
        self.max_dR = load_dict["max_dR"]
        return self


class UnbinnedCalibration:
    def __init__(
        self,
        max_dR=0.3,
        hyperparams={"n_estimators": 100, "max_depth": 3, "early_stopping_rounds": 5},
    ):
        self.bdt = xgb.XGBRegressor(**hyperparams)
        self.max_dR = max_dR

    def fit(self, truth_jets, recon_jets):
        matched_recon_jets, matched_truth_jets = truth_match(
            recon_jets, truth_jets, max_dR=self.max_dR
        )

        truth_pT = ak.to_numpy(ak.flatten(matched_truth_jets.pt))
        recon_pT = ak.to_numpy(ak.flatten(matched_recon_jets.pt))
        recon_abseta = np.abs(ak.to_numpy(ak.flatten(matched_recon_jets.eta)))

        X = np.stack((recon_pT, recon_abseta), axis=1)
        y = truth_pT / recon_pT

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.bdt.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        return self

    def predict(self, recon_jets):
        recon_pT = ak.to_numpy(ak.flatten(recon_jets.pt))
        recon_abseta = np.abs(ak.to_numpy(ak.flatten(recon_jets.eta)))

        X = np.stack((recon_pT, recon_abseta), axis=1)

        if isinstance(self.bdt, xgb.Booster):
            X = xgb.DMatrix(X)

        pred_sf = ak.unflatten(self.bdt.predict(X), ak.num(recon_jets))

        scaled_recon_jets = {
            "m": recon_jets.m,
            "pt": recon_jets.pt * pred_sf,
            "eta": recon_jets.eta,
            "phi": recon_jets.phi,
        }

        return vector.zip(scaled_recon_jets)

    def save(self, path):
        self.bdt.save_model(path)

    def load(self, path):
        self.bdt = xgb.Booster()
        self.bdt.load_model(path)
        return self
