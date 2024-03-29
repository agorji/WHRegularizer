import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from utils import ModelTrainer
from datasets import FourierDataset

class FCN(nn.Module):
    def __init__(self, n, multiplier=2, batch_norm=False):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(n, multiplier*n)
        self.fc2 = nn.Linear(multiplier*n, multiplier*n)
        self.fc3 = nn.Linear(multiplier*n, n)
        self.fc4 = nn.Linear(n, 1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(multiplier*n)
            self.bn2 = nn.BatchNorm1d(multiplier*n)
            self.bn3 = nn.BatchNorm1d(n)

    def forward(self, x):
        if self.batch_norm:
            x = self.bn1(F.leaky_relu(self.fc1(x)))
            x = self.bn2(F.leaky_relu(self.fc2(x)))
            x = self.bn3(F.leaky_relu(self.fc3(x)))
            x = self.fc4(x)
        else:
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = F.leaky_relu(self.fc3(x))
            x = self.fc4(x)

        return x.reshape(-1)

def main():  
    run = wandb.init()
    print(wandb.config)

    n = wandb.config.n
    k = wandb.config.k
    d = wandb.config.d
    config = wandb.config

    # Dataset
    config["train_size"] = config["dataset_size_coef"] * k * n
    config["val_size"] = config["train_size"] * 5
    dataset_size = config["train_size"] + config["val_size"] * 2

    dataset_args = {"random_seed": config["fix_seed"] if config.get("fix_dataset", False) else config["random_seed"]}
    if "freq_sampling_method" in config:
        dataset_args["freq_sampling_method"] = config["freq_sampling_method"]
    if "sample_seed" in config:
        dataset_args["freq_seed"] = config["fix_seed"]
        dataset_args["random_seed"] = config["sample_seed"]

    dataset = FourierDataset(n, k, d=d, n_samples=dataset_size, **dataset_args)
    if config.get("normalize_data", False):
        dataset.y = (dataset.y-torch.mean(dataset.y))/torch.std(dataset.y)

    train_ds = torch.utils.data.Subset(dataset, list(range(config["train_size"])))
    val_ds = torch.utils.data.Subset(dataset, list(range(config["train_size"], config["train_size"]+config["val_size"])))
    test_ds = torch.utils.data.Subset(dataset, list(range(config["train_size"]+config["val_size"], dataset_size)))

    # Train model
    remove_from_train_config = ["dataset_size_coef", "val_size"]
    train_config = {k:v for k, v in config.items() if k not in remove_from_train_config}

    torch.manual_seed(config["random_seed"]) # Seed for network initialization
    in_dim = dataset.X.shape[1]
    model = FCN(in_dim, config["network_c"])
    trainer = ModelTrainer(model, train_ds, val_ds, config=train_config, log_wandb=True, checkpoint_cache=True, 
                            experiment_name="synthetic", test_ds=test_ds)
    model = trainer.train_model()

if __name__ == "__main__":
    main()
