import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from utils import ModelTrainer
from datasets import avGFPDataset, GB1Dataset, SGEMMDataset, EntacmaeaDataset

class FCN(nn.Module):
    def __init__(self, n, width_coeff=2, batch_norm=False):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(n, width_coeff*n)
        self.fc2 = nn.Linear(width_coeff*n, width_coeff*n)
        self.fc3 = nn.Linear(width_coeff*n, n)
        self.fc4 = nn.Linear(n, 1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(width_coeff*n)
            self.bn2 = nn.BatchNorm1d(width_coeff*n)
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
    config = wandb.config

    # Dataset
    if config["dataset"] == "GB1":
        dataset = GB1Dataset()
    elif config["dataset"] == "avGFP":
        dataset = avGFPDataset()
    elif config["dataset"] == "SGEMM":
        dataset = SGEMMDataset()
    elif config["dataset"] == "Entacmaea":
        dataset = EntacmaeaDataset()
    else:
        raise Exception
    if config.get("normalize_data", False):
        dataset.y = (dataset.y-torch.mean(dataset.y))/torch.std(dataset.y)
    
    torch.manual_seed(config["fix_seed"])
    config["val_size"] = config.get("val_size", len(dataset) - config["train_size"])
    remainder = len(dataset) - config["train_size"] - config["val_size"]
    train_ds, val_ds, _ = torch.utils.data.random_split(dataset, [config["train_size"], config["val_size"], remainder])

    # Train model
    remove_from_train_config = ["val_size"]
    train_config = {k:v for k, v in config.items() if k not in remove_from_train_config}

    torch.manual_seed(config["random_seed"]) # Seed for network initialization
    in_dim = dataset.X.shape[1]
    model = FCN(in_dim, config["network_c"], batch_norm=config.get("batch_norm", False))
    trainer = ModelTrainer(model, train_ds, val_ds, config=train_config, log_wandb=True, checkpoint_cache=True, 
                            experiment_name=config["dataset"])
    model = trainer.train_model()

if __name__ == "__main__":
    main()
