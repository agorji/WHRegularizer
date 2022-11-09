import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from utils import FourierDataset, ModelTrainer

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
    d = math.ceil(n/3)
    config = wandb.config
    config["b"] = math.ceil(math.log2(k)) + config.get("hashing_discount", 0)

    # Set seeds
    random_seed = config["random_seed"]
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Dataset
    dataset_size = config["dataset_size_coef"] * k * n
    dataset = FourierDataset(n, k, d=d, n_samples=int(dataset_size*(4/3)))

    # Set batch size
    config["epoch_iterations"] = config["SPRIGHT_d"]
    batch_size = dataset_size / config["epoch_iterations"]
    a = 1
    while(a < batch_size):
        a *= 2
    config["batch_size"] = a

    # Train model
    in_dim = dataset.X.shape[1]
    model = FCN(in_dim, 2)
    trainer = ModelTrainer(model, dataset, config=config, log_wandb=True, report_epoch_fourier=True)
    spectrums = trainer.train_model()

if __name__ == "__main__":
    main()
