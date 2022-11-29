import numpy as np
import os
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).parent / "data"

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


if __name__ == "__main__":
    ds = GB1Dataset()
    print(ds.X.shape)
    print(ds.X[:5])
    print(ds.y[:5])

    ds = avGFPDataset()
    print(ds.X.shape)
    print(ds.X[:5])
    print(ds.y[:5])