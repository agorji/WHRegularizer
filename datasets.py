import numpy as np
import os
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).parent / "data"

def get_real_dataset(dataset_name):
    if dataset_name == "GB1":
        return GB1Dataset()
    elif dataset_name == "avGFP":
        return avGFPDataset()
    elif dataset_name == "SGEMM":
        return SGEMMDataset()
    elif dataset_name == "Entacmaea":
        return EntacmaeaDataset()
    else:
        raise Exception(f"{dataset_name} is not among available real datasets.")

class FourierDataset(Dataset):
    def __init__(self, n, k, freq_sampling_method="uniform_deg", amp_sampling_method="random", d=None, p_freq=None, n_samples = 100, p_t=0.5, 
                random_seed=0, use_cache=True, freq_seed=None):
        self.n = n
        self.k = k
        self.d = d
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.freq_seed = freq_seed
        self.freq_sampling_method = freq_sampling_method
        self.amp_sampling_method = amp_sampling_method

        # Check if n_samples makes sense
        if n_samples > 2**self.n:
            raise Exception(f"Can't generate a dataset of size {n_samples} for a space with n={self.n}")

        # Load from cache if possible
        if use_cache:
            if self.load_from_cache():
                return

        self.generator = torch.Generator()
        self.generator.manual_seed(freq_seed if freq_seed is not None else random_seed)

        # Freqs
        if freq_sampling_method == "uniform_deg":
            self.freq_f = self.uniform_deg_freq(d)
        elif freq_sampling_method == "fixed_deg":
            self.freq_f = self.fixed_deg_freq(d)
        elif freq_sampling_method == "single_deg":
            if self.k > self.n:
                raise Exception("k cannot be bigger than n in fixed_deg.")
            self.freq_f = self.single_deg_freq()
        elif freq_sampling_method == "bernouli":
            self.freq_f = self.bernouli_freq(p_freq)
        else:
            raise Exception(f'"{freq_sampling_method}" is not a generation method supported in FourierDataset.')

        # Amplitudes
        if amp_sampling_method == "random":
            self.amp_f = torch.FloatTensor(k).uniform_(-1, 1, generator=self.generator)
        elif amp_sampling_method == "constant":
            self.amp_f = torch.ones(k)
        
        # If freq_seed is given we reset random seed
        if freq_seed is not None:
            self.generator.manual_seed(random_seed)

        # Data
        self.X = (torch.rand(n_samples, n, generator=self.generator) < p_t).float()
        ## --- deduplicating data
        self.X = torch.unique(self.X, dim=0)
        while self.X.shape[0] < n_samples:
            self.X = torch.vstack([self.X, (torch.rand(n_samples - self.X.shape[0], n, generator=self.generator) < p_t).float()])
            self.X = torch.unique(self.X, dim=0)
        
        self.X = self.X[torch.randperm(n_samples)]
        self.y = self.compute_y(self.X)

        if use_cache:
            self.cache()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def compute_y(self, X):
        t_dot_f = X @ torch.t(self.freq_f)
        return torch.sum(torch.where(t_dot_f % 2 == 1, -1, 1) * self.amp_f, axis = -1) / self.k

    def bernouli_freq(self, p):
        return (torch.rand(self.k, self.n, generator=self.generator) < p).float()

    def uniform_deg_freq(self, d):
        freqs = []
        weights = torch.ones(self.n)
        for i in range(self.k):
            deg = torch.randint(1, d+1, (1,),  generator=self.generator).item()
            is_duplicate = True
            while is_duplicate:
                one_indices = torch.multinomial(weights, deg, generator=self.generator)
                new_f = torch.zeros(self.n).float()
                new_f[one_indices] = 1.0
                new_f = new_f.tolist()

                if new_f not in freqs:
                    freqs.append(new_f)
                    is_duplicate = False
        return torch.tensor(freqs).float()

    def single_deg_freq(self):
        freqs = []
        weights = torch.ones(self.n)
        for i in range(self.k):
            deg = i + 1

            one_indices = torch.multinomial(weights, deg, generator=self.generator)
            new_f = torch.zeros(self.n).float()
            new_f[one_indices] = 1.0
            new_f = new_f.tolist()
            freqs.append(new_f)

        return torch.tensor(freqs).float()

    def fixed_deg_freq(self, d):
        freqs = []
        weights = torch.ones(self.n)
        deg = d
        for i in range(self.k):
            is_duplicate = True
            while is_duplicate:
                one_indices = torch.multinomial(weights, deg, generator=self.generator)
                new_f = torch.zeros(self.n).float()
                new_f[one_indices] = 1.0
                new_f = new_f.tolist()

                if new_f not in freqs:
                    freqs.append(new_f)
                    is_duplicate = False
        return torch.tensor(freqs).float()
    
    def get_cache_dir(self):
        data_directory = os.environ.get("EXPERIMENT_DATA") if "EXPERIMENT_DATA" in os.environ else os.getcwd()

        dataset_dir = f"{data_directory}/datasets/n{self.n}_k{self.k}_d{self.d}/"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        return dataset_dir
    
    def get_cache_file_name(self):
        freq_seed_postfix = f"(freq{self.freq_seed})" if self.freq_seed is not None else ""
        return f'{self.n_samples}_seed{self.random_seed}{freq_seed_postfix}_{self.freq_sampling_method}_{self.amp_sampling_method}.pth'
    
    def cache(self):
        model_dir = self.get_cache_dir()
        cache_file_name = self.get_cache_file_name()

        data = { 
                'X': self.X,
                'y': self.y,
                'freqs': self.freq_f,
                'amps': self.amp_f,
            }
        torch.save(data, f'{model_dir}/{cache_file_name}')
    
    def load_from_cache(self):
        model_dir = self.get_cache_dir()
        cache_file_name = self.get_cache_file_name()
        dataset_file = f'{model_dir}/{cache_file_name}'
        if os.path.exists(dataset_file):
            loaded_data = torch.load(dataset_file)
            self.X = loaded_data["X"]
            self.y = loaded_data["y"]
            self.freq_f = loaded_data["freqs"]
            self.amp_f = loaded_data["amps"]
            return True
        else:
            return False

    def get_int_freqs(self):
        return self.freq_f.cpu().numpy().dot(2**np.arange(self.n)[::-1]).astype(int)

    def get_fourier_spectrum(self):
        spectrum = np.zeros(2**self.n)
        spectrum[self.get_int_freqs()] = self.amp_f.cpu().numpy()
        return spectrum


class GB1Dataset(Dataset):
    def __init__(self, use_cache=True):
        # Load from storage if requested and available
        if use_cache:
            cache_file = f"{DATA_PATH}/gb1.pt"
            if os.path.exists(cache_file):
                self.X, self.y = torch.load(cache_file)
                print("Loaded dataset from cache.")
                return

        # Load data
        data_file = f"{DATA_PATH}/gb1-1.csv"
        if os.path.exists(data_file):
            data_df = pd.read_csv(data_file)
        else:
            raise Exception(f"Could not find GB1 data in '{data_file}'")
        
        self.variants = data_df["Variants"]
        self.X = torch.Tensor(self.one_hot_encode_variants()).float()
        self.y = torch.Tensor(data_df["Fitness"])

        if use_cache:
            torch.save((self.X, self.y), cache_file)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def one_hot_encode_variants(self):
        # Check if all variants have the same length
        lengths = [len(v) for v in self.variants]
        assert min(lengths) == max(lengths)

        # Split each variants into its list of characters to feed into one hot encoder
        splitted_variants = [[c for c in v] for v in self.variants]
        return OneHotEncoder(sparse=False).fit_transform(splitted_variants)


class avGFPDataset(Dataset):
    def __init__(self, use_cache=True):
        # Load from storage if requested and available
        if use_cache:
            cache_file = f"{DATA_PATH}/avGFP.pt"
            if os.path.exists(cache_file):
                self.X, self.y = torch.load(cache_file)
                print("Loaded dataset from cache.")
                return
        
        # Load data
        data_file = f"{DATA_PATH}/avGFP.csv"
        if os.path.exists(data_file):
            self.data_df = pd.read_csv("data/avGFP.csv", delimiter="\t", keep_default_na=False)
        else:
            raise Exception(f"Could not find avGFP data in '{data_file}'")
        
        self.X, self.y = self.compute_mutation_dataset()
        
        if use_cache:
            torch.save((self.X, self.y), cache_file)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def compute_mutation_dataset(self):
        mutations = self.data_df["aaMutations"][1:] # Ignore the first row which is the reference

        # Convert mutations into tuples of site indices
        splited_mutations = [s.split(":") for s in mutations]
        mutation_sites = [()] + [tuple(sorted([int(aam[2:-1]) for aam in aams])) for aams in splited_mutations]
        self.data_df["mutation_sites"] = mutation_sites

        # Aggregate similar records (same mutation sites but different mutations) into their average brightness
        aggregated_df = self.data_df.groupby('mutation_sites').agg({'medianBrightness': np.mean}).reset_index()
        aggregated_df.columns = aggregated_df.columns.get_level_values(0)
        
        # Generate X, y
        mutation_sites = [list(aam_tup) for aam_tup in aggregated_df["mutation_sites"]][1:]
        site_count = max([max(l) for l in mutation_sites if len(l) > 0]) + 1

        X = np.zeros((len(mutation_sites), site_count))
        X[np.arange(X.shape[0]).repeat([*map(len, mutation_sites)]), np.concatenate(mutation_sites)] = 1
        X = np.vstack([np.zeros([1, site_count]), X]) # Add the reference to X

        return torch.from_numpy(X).float(), torch.Tensor(aggregated_df["medianBrightness"]).float()

class EntacmaeaDataset(Dataset):
    def __init__(self, use_cache=True):
        # Load from storage if requested and available
        if use_cache:
            cache_file = f"{DATA_PATH}/quadricolor_fluorscent.pt"
            if os.path.exists(cache_file):
                self.X, self.y = torch.load(cache_file)
                print("Loaded dataset from cache.")
                return

        # Load data
        data_file = f"{DATA_PATH}/quadricolor_fluorscent.csv"
        if os.path.exists(data_file):
            data_df = pd.read_csv(data_file)
        else:
            raise Exception(f"Could not find GB1 data in '{data_file}'")
        
        self.mutations = data_df["genotype"]
        # Convert input binary strings to tensors
        self.mutations = [[int(bit) for bit in m[1:-1]]
                            for m in self.mutations]
        self.X = torch.Tensor(self.mutations).float()
        self.y = torch.Tensor(data_df["brightness"]).float()

        if use_cache:
            torch.save((self.X, self.y), cache_file)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class SGEMMDataset(Dataset):
    def __init__(self, use_cache=True):
        # Load from storage if requested and available
        if use_cache:
            cache_file = f"{DATA_PATH}/sgemm.pt"
            if os.path.exists(cache_file):
                self.X, self.y = torch.load(cache_file)
                print("Loaded dataset from cache.")
                return

        # Load data
        data_file = f"{DATA_PATH}/sgemm.csv"
        if os.path.exists(data_file):
            data_df = pd.read_csv(data_file)
        else:
            raise Exception(f"Could not find SGEMM data in '{data_file}'")
        
        # Aggregate runtimes
        runtime_columns = ["Run1 (ms)","Run2 (ms)","Run3 (ms)","Run4 (ms)"]
        data_df["time"] = np.mean(data_df[runtime_columns], axis=1)
        data_df = data_df.drop(columns=runtime_columns)

        self.X = data_df.drop(columns=["time"])
        self._one_hot_transform()
        self.X = torch.Tensor(self.X).float()
        self.y = torch.Tensor(data_df["time"]).float()

        if use_cache:
            torch.save((self.X, self.y), cache_file)
        
    def _one_hot_transform(self):
        self.enc = OneHotEncoder(sparse=False)
        self.X = self.enc.fit_transform(self.X)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

if __name__ == "__main__":
    ds = GB1Dataset()
    print(ds.X.shape)
    print(ds.X[:5])
    print(ds.y[:5])

    ds = avGFPDataset()
    print(ds.X.shape)
    print(ds.X[:5])
    print(ds.y[:5])

    ds = EntacmaeaDataset()
    print(ds.X.shape)
    print(ds.X[:5])
    print(ds.y[:5])

    ds = SGEMMDataset()
    print(ds.X.shape)
    print(ds.X[:5])
    print(ds.y[:5])