import vector
from fastml.utils_egamma.efex import *
from fastml.utils.misc import get_config
from fastml.utils.image import vector_to_tower, pad, sliding_window
from fastml.utils_egamma.misc import get_open_data

from data.opendatahelper import drop_overlapping

class OpenDataLoader:
    def __init__(self, 
                 sample_name):
        self.sample_name = sample_name 
        
        self.towers = None 
        self.towers_noPU = None

        self.seed_vectors = None 
        self.truth_vectors = None
        
        self.weight = None 

    def load(self, n_start = None, n_stop = None):
        config = get_config(self.sample_name)
        for no_PU in [True, False]:

            data = get_open_data(
                config,
                n_start=n_start,
                n_stop=n_stop,
                no_PU=no_PU
            )

            mask = abs(data["Tower/Tower.Eta"]) <= 2.5

            vec_eem = vector.zip({
                "pt": data["Tower/Tower.Eem"][mask],
                "eta": data["Tower/Tower.Eta"][mask],
                "phi": data["Tower/Tower.Phi"][mask],
                "m": ak.zeros_like(data["Tower/Tower.Eem"][mask]),
            })

            vec_ehad = vector.zip({
                "pt": data["Tower/Tower.Ehad"][mask],
                "eta": data["Tower/Tower.Eta"][mask],
                "phi": data["Tower/Tower.Phi"][mask],
                "m": ak.zeros_like(data["Tower/Tower.Ehad"][mask]),
            })

            towers_eem = np.asarray(vector_to_tower(vec_eem))
            towers_ehad = np.asarray(vector_to_tower(vec_ehad))

            towers = np.concatenate([towers_eem, towers_ehad], axis=-1)

            if no_PU:
                self.towers_noPU = towers
            else:
                self.towers = towers

        self._load_seeddata()
        self.weight = data['Event/Event.Weight']

    def _load_seeddata(self):
        if 'Zee' in self.sample_name:
            gev_mask = self.towers_noPU[..., 0] > 10
        else:
            gev_mask = self.towers[..., 0] > 10
        t = np.argwhere(gev_mask)
        ev_ids = t[:,0]
        e0 = t[:,1]
        p0 = t[:,2]

        eta_edges=np.linspace(-2.5, 2.5, 51)
        phi_edges=np.linspace(-np.pi, np.pi, 65)

        eta = np.tile(eta_edges, (len(self.towers), 1))[ev_ids, e0] 
        phi = np.tile(phi_edges, (len(self.towers), 1))[ev_ids, p0] 
        pt = self.towers[ev_ids, e0, p0, 0]

        sorted_idx = np.argsort(ev_ids)
        counts = np.bincount(ev_ids[sorted_idx], minlength=len(self.towers))

        eta = ak.unflatten(eta[sorted_idx], counts)
        phi = ak.unflatten(phi[sorted_idx], counts)
        pt = ak.unflatten(pt[sorted_idx], counts)

        seed_vectors = vector.zip({
            "pt": pt,  # GeV
            "eta": eta,
            "phi": phi,
            "m": 0,
        })

        seed_vectors_mask = self._eta_pt_mask(seed_vectors, min_pt = 10)
        seed_vectors = seed_vectors[seed_vectors_mask]
        print(sum(seed_vectors.rho.layout.content))
        exit()
        seed_vectors.show()
        print(sum(seed_vectors.rho))
        print(sum(sum(seed_vectors.rho)))
        self.seed_vectors = seed_vectors[drop_overlapping(seed_vectors)]
        self.seed_vectors.show()
        exit()
        if 'Zee' in self.sample_name:
            truth_pt = self.towers_noPU[ev_ids, e0, p0, 0]
            truth_pt = ak.unflatten(truth_pt[sorted_idx], counts)
            truth_vectors = self._load_truthdata(truth_pt, eta, phi)
            truth_vectors = truth_vectors[seed_vectors_mask]
            self.truth_vectors = truth_vectors[drop_overlapping(truth_vectors)]

    def _load_truthdata(self, truth_pt, eta, phi):
        return vector.zip({
            "pt": truth_pt,  # GeV
            "eta": eta,
            "phi": phi,
            "m": 0,
        })
                
    def _eta_pt_mask(self, obj, min_pt = 0.0):
        abs_eta = np.abs(obj.eta)
        return ((abs_eta < 2.5) & (obj.pt > min_pt) & ((abs_eta < 1.37) | (abs_eta > 1.52)))
    
    def save_to_parquet(self, save_path):
        _, (ev_ids, e0, p0) = eFex_slidingwindow_mask(self.seed_vectors)
        towers = sliding_window(pad(self.towers, 1), 3)[ev_ids, e0, p0]
        towers_ak = ak.unflatten(towers, ak.num(self.seed_vectors, axis=1))

        output = ak.zip(
            {
                "image": towers_ak,
                "seed_info": self.seed_vectors,
            },
            depth_limit=1,
        )

        ak.to_parquet(output, save_path)


    
