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
from wandb_utils import get_wandb_logs

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
    config = wandb.config
    config["k"] = n
    config["b"] = math.ceil(math.log2(config["k"])) + config.get("hashing_discount", 0)

    # Dataset params
    config["train_size"] = math.ceil(config["dataset_size_coef"] * config["k"] * n)
    config["val_size"] =  2**n - config["train_size"]
    dataset_size = 2**n
    random_seed = config["fix_seed"] if config.get("fix_dataset", False) else config["random_seed"]
    if config["train_size"] >= dataset_size:
        raise Exception("Impossible (large) training size given:", config["train_size"])

    # Dataset
    dataset = FourierDataset(n, config["k"], d=config["d"], n_samples=dataset_size, amp_sampling_method="constant", random_seed=random_seed)
    train_ds = torch.utils.data.Subset(dataset, list(range(config["train_size"])))
    val_ds = torch.utils.data.Subset(dataset, list(range(config["train_size"], dataset_size)))

    # Train model
    remove_from_train_config = ["hashing_discount", "dataset_size_coef", "val_size"]
    train_config = {k:v for k, v in config.items() if k not in remove_from_train_config}

    torch.manual_seed(config["random_seed"]) # Seed for network initialization
    in_dim = dataset.X.shape[1]
    model = FCN(in_dim, 10)
    args = {"data_freqs": dataset.get_int_freqs(), "data_spectrum": dataset.get_fourier_spectrum() * (2**(n/2) / config["k"])}
    trainer = ModelTrainer(model, train_ds, val_ds, config=train_config, log_wandb=True, report_epoch_fourier=True, 
                            experiment_name="spectrum", checkpoint_interval=100, **args)
    spectrums = trainer.train_model()

if __name__ == "__main__":
    main()
