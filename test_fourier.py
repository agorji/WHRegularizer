import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np

from itertools import product
import math
import random
import wandb

class FourierDataset(Dataset):
    def __init__(self, n, k, generation_method="uniform_deg", d=None, p_freq=None, n_samples = 100, p_in=0.5):
        self.n = n
        self.k = k

        if generation_method == "uniform_deg":
            self.freq_f = self.uniform_deg_freq(d)
        elif generation_method == "bernouli":
            self.freq_f = self.bernouli_freq(p_freq)
        elif generation_method == "bounded_deg":
            self.freq_f = self.bounded_degree_freq(d)
        else:
            raise Exception(f'"{generation_method}" is not a generation method for FourierDataset.')

        self.amp_f = torch.FloatTensor(k).uniform_(-1, 1)

        self.X = (torch.rand(n_samples, n) < p_in).float()
        self.y = self.compute_y(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def compute_y(self, X):
        in_dot_f = X @ torch.t(self.freq_f)
        return torch.sum(torch.where(in_dot_f % 2 == 1, -1, 1) * self.amp_f, axis = -1) / 2**(self.n/2)
    
    def bounded_degree_freq(self, d):
        freqs = torch.zeros(self.k, self.n)
        one_indices = [(i, random.randrange(self.n)) for j in range(d) for i in range(self.k)]
        freqs[list(zip(*one_indices))] = 1.0
        return freqs

    def bernouli_freq(self, p):
        return (torch.rand(self.k, self.n) < p).float()

    def uniform_deg_freq(self, d):
        freqs = torch.zeros(self.k, self.n)
        one_rows, one_cols = [], []
        for i in range(self.k):
            deg = random.randint(1, d)
            one_rows.extend([i]*deg)
            one_cols.extend(random.sample(range(self.n), deg))
        freqs[one_rows, one_cols] = 1.0
        return freqs

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

def hadamard_matrix(n, normalize=False):
    '''
    Compute H_n, Hadamard matrix
    '''
    H1 = torch.asarray([[1.,1.], [1.,-1.]])
    H = torch.asarray([1.])
    for i in range(n):
        H = torch.kron(H, H1)
    if normalize:
        H = (1 / math.sqrt(2**n)) * H
    return H

def get_sample_inputs(n, b):
    hash_sigma = (torch.rand(b, n) < 0.5).float() # multivariate bernouli with p=0.5
    hash_inputs = torch.asarray(list((product((0.0,1.0), repeat=b))))

    sample_inputs = hash_inputs @ hash_sigma
    return sample_inputs

    
def train_model(model, train_loader, val_loader, config, device="cuda"):
    if config["add_hadamard_loss"]:
        H = hadamard_matrix(config["b"], normalize=True).to(device)
        n = next(iter(val_loader))[0].shape[1] # original space dimension

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    for epoch in range(config["num_epochs"]):
        model.train()
        batch_train_loss, batch_hadamard_loss = [], []
        for X, y in train_loader:
            optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = F.mse_loss(y, y_hat)
            batch_train_loss.append(loss)

            if config["add_hadamard_loss"]:
                # Find the sample inputs using the hashing scheme
                sample_inputs = get_sample_inputs(n, config["b"])
                sample_inputs = sample_inputs.to(device)

                # Compute the Hadamard transform of sample_inputs and add to loss
                X = model(sample_inputs)
                Y = H @ X

                hadamard_loss = F.l1_loss(Y, torch.zeros_like(Y))
                batch_hadamard_loss.append(hadamard_loss)
                loss += config["hadamard_lambda"] * hadamard_loss

            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                val_loss += F.mse_loss(y, y_hat).item()
            val_loss /= len(val_loader)
            print(f"#{epoch} - Validation Loss: {val_loss:.3f}")

        # Log wandb
        wandb.log({
            'val_loss': val_loss,
            'train_loss': torch.mean(torch.stack(batch_train_loss)).item(),
            'hadamard_loss': -1.0 if len(batch_hadamard_loss) == 0 else torch.mean(torch.stack(batch_hadamard_loss)).item(),
        })
    return model

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

    # Dataloader
    p_test = 0.25

    test_size = int(p_test * len(dataset))
    train_ds, val_ds = random_split(dataset, lengths=[len(dataset) - test_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=wandb.config.batch_size, shuffle=True)

    # Train model
    in_dim = dataset.X.shape[1]
    model = FCN(in_dim, 2)
    model = train_model(model, train_loader, val_loader, config=wandb.config)

if __name__ == "__main__":
    # Start sweep job.
    # wandb.agent(sweep_id, function=main)
    main()
