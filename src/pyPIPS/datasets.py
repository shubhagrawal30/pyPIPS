import numpy as np
import pyDOE
import tqdm
import pickle

class Datapoint():
    def __init__(self, ks, zs, P_kz, params):
        assert len(ks) == P_kz.shape[0], "first index of P_kz must be k's"
        assert len(zs) == P_kz.shape[1], "second index of P_kz must be k's"
        self.ks = ks
        self.zs = zs
        self.P_kz = P_kz
        self.params = params

class Dataset():
    def __init__(self):
        self.datapoints = None
        self.P_kzs = None
        self.ks = None
        self.zs = None
        self.num_points = 0
        self.parameters = None
        self.all_parameters = None
    
    def add(self, datapoint: Datapoint):
        if self.parameters is None:
            self.parameters = list(datapoint.params.keys())
        else:
            assert np.array_equal(self.parameters, list(datapoint.params.keys())), "parameters must be the same for all datapoints"
        if self.datapoints is None:
            self.datapoints = []
            self.P_kzs = np.empty((0, len(datapoint.ks), len(datapoint.zs)))
            self.ks = datapoint.ks
            self.zs = datapoint.zs
            self.all_parameters = np.empty((0, len(self.parameters)))
        else:
            assert np.array_equal(self.ks, datapoint.ks), "ks must be the same for all datapoints"
            assert np.array_equal(self.zs, datapoint.zs), "zs must be the same for all datapoints"
        self.datapoints.append(datapoint)
        self.P_kzs = np.append(self.P_kzs, datapoint.P_kz[None], axis=0)
        self.all_parameters = np.append(self.all_parameters, \
                                np.array(list(datapoint.params.values()))[None], axis=0)
        self.num_points += 1
    
    def generate_from_func(self, func, num_points, parameter_range, sampling="LH"):
        if self.parameters is None:
            self.parameters = list(parameter_range.keys())
        par_range = np.array(list(parameter_range.values()))
        if sampling == "LH":
            lhs_sample = pyDOE.lhs(len(self.parameters), samples=num_points, criterion='c')
            lhs_cosmology = lhs_sample * (par_range[:,1] - par_range[:,0]) + par_range[:,0]
        
        for i in tqdm.tqdm(range(num_points)):
            params = dict(zip(self.parameters, lhs_cosmology[i]))
            self.add(func(params))

    def get_datapoint(self, i):
        return self.datapoints[i]
    
    def get_P_kzs(self):
        return self.P_kzs
    
    def get_parameters(self):
        return self.parameters
    
    def get_all_parameters(self):
        return self.all_parameters
    
    def get_P_kz(self, i):
        return self.datapoints[i].P_kz

    def get_num_points(self):
        return self.num_points

    def get_ks(self):
        return self.ks
    
    def get_zs(self):
        return self.zs
    
    def split(self, ratios):
        assert np.sum(ratios) == 1, "ratios must sum to 1"
        num_points = self.get_num_points()
        split_points = np.cumsum(np.array(ratios) * num_points).astype(int)
        split_points = np.insert(split_points, 0, 0)
        datasets = []
        for i in range(len(split_points) - 1):
            datasets.append(Dataset())
            for j in tqdm.tqdm(range(split_points[i], split_points[i+1])):
                datasets[i].add(self.get_datapoint(j))
        return datasets
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
