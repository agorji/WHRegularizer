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

# Define sweep config
sweep_configuration = {
    'method': 'grid',
    'name': 'fixed_fourier_first',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': {
        'n': {'values': [16, 48, 128]},
        'k': {'values': [10, 40, 100]},
        'b': {'values': [4, 8, 12]},
        'add_hadamard_loss': {'values': [True]}, 
        # 'lr': {'max': 0.1, 'min': 0.0001},
        'lr': {'values': [0.1, 0.01, 0.001, 0.0001]}, 
        'weight_decay': {'values': [0.0]}, 
        'hadamard_lambda': {'values': [0.0, 0.1, 0.5, 1.0, 2.0]},
        'num_epochs': {'values': [50]},
        'batch_size': {'values': [64]},
        'dataset_size': {'values': [100, 500, 1000, 10000]},
    }
}

# Initialize sweep by passing in config.
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='SpectralRegularizer')

def main():  
    run = wandb.init()
    print(wandb.config)

    n = wandb.config.n
    k = wandb.config.k

    # Set seeds
    random_seed = 0
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Dataset
    dataset = FourierDataset(n, k, d=n/8, n_samples=wandb.config.dataset_size)

    # Train model
    in_dim = dataset.X.shape[1]
    model = FCN(in_dim, 2)
    trainer = ModelTrainer(model, dataset, training_method="hashing", config=wandb.config)
    model = trainer.train_model()

if __name__ == "__main__":
    # Start sweep job.
    # wandb.agent(sweep_id, function=main)
    main()
