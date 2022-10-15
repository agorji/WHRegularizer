import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List

from itertools import product
import math
import random
import wandb

from epistatic_net.spright_utils import SPRIGHT, make_system_simple
from epistatic_net.wht_sampling import SPRIGHTSample

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


class ModelTrainer:
    def __init__(self, model: nn.Module, dataset: Dataset, config: Dict, training_method: str,
                p_test = 0.25, device = "cuda", log_wandb = False):
        '''
            Trains a torch model given the dataset and the training method.

            Args:
                - model: the torch model to train
                - dataset: the torch dataset to use for training
                - training_method: training method to use for training which usually sets how the loss should
                                be calculated in each epoch
                - config: a config dictionary that includes all the hyperparameters for training
                - p_test: relative size of the validation dataset
                - device: torch device used for the training
        '''

        self.device = device
        self.model = model
        self.config = config
        self.log_wandb = log_wandb
        self.dataset = dataset

        # Dataloader
        test_size = int(p_test * len(self.dataset))
        train_ds, val_ds = random_split(self.dataset, lengths=[len(self.dataset) - test_size, test_size])

        self.train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=True)
        self.n = next(iter(self.val_loader))[0].shape[1] # original space dimension

        # Training method
        if training_method == "normal":
            self.train_epoch = self.normal_epoch

        elif training_method == "EN":
            self.train_epoch = self.EN_epoch
            self.all_inputs = torch.asarray(list((product((0,1), repeat=self.n)))).to(device)
            self.H = hadamard_matrix(self.n, normalize=True).to(device)

        elif training_method == "EN-S":
            self.train_epoch = self.ENS_epoch
            self.spright_sample = SPRIGHTSample(self.n, config["SPRIGHT_m"], config["SPRIGHT_d"])
            self.X_all = np.concatenate(
                (
                    np.vstack(self.spright_sample.sampling_locations[0]),
                    np.vstack(self.spright_sample.sampling_locations[1]),
                    np.vstack(self.spright_sample.sampling_locations[2])
                )
            )
            self.X_all, self.X_all_inverse_ind = np.unique(self.X_all, axis=0, return_inverse='True')

            # initialzie ADMM 
            self.Hu = np.zeros(len(self.X_all))
            self.lam = np.zeros(len(self.X_all))

        elif training_method == "hashing":
            self.train_epoch = self.hashing_epoch
            self.H = hadamard_matrix(config["b"], normalize=True).to(device)
            
        else:
            raise Exception(f"'{training_method}' training method is not yet implemented.")

        # Training stuff
        self.optim = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    def train_model(self):
        device = self.device

        self.model.to(device)
        for epoch in range(self.config["num_epochs"]):
            # Train epoch based on the set training method
            self.model.train()
            epoch_log = self.train_epoch()

            # Evaluate the model on validation set
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for X, y in self.val_loader:
                    X, y = X.to(device), y.to(device)
                    y_hat = self.model(X)
                    val_loss += F.mse_loss(y, y_hat).item()
                val_loss /= len(self.val_loader)
                print(f"#{epoch} - Validation Loss: {val_loss:.3f}")
            
            epoch_log['val_loss']  = val_loss

            # Log wandb
            if self.log_wandb:
                wandb.log(epoch_log)
    
    def normal_epoch(self):
        device = self.device
        batch_train_loss = []
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            batch_train_loss.append(loss)

            loss.backward()
            self.optim.step()
        
        return {
            'train_loss': torch.mean(torch.stack(batch_train_loss)).item(),
        }
    
    def hashing_epoch(self):
        device = self.device
        batch_train_loss, batch_hadamard_loss = [], []
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            batch_train_loss.append(loss)

            # Find the sample inputs using the hashing scheme
            sample_inputs = get_sample_inputs(self.n, self.config["b"])
            sample_inputs = sample_inputs.to(device)

            # Compute the Hadamard transform of sample_inputs and add to loss
            X = self.model(sample_inputs)
            Y = self.H @ X

            hadamard_loss = F.l1_loss(Y, torch.zeros_like(Y))
            batch_hadamard_loss.append(hadamard_loss)
            loss += self.config["hadamard_lambda"] * hadamard_loss

            loss.backward()
            self.optim.step()
        
        return {
            'train_loss': torch.mean(torch.stack(batch_train_loss)).item(),
            'hashing_loss': torch.mean(torch.stack(batch_hadamard_loss)).item(),
        }
    
    def EN_epoch(self):
        device = self.device
        batch_train_loss, batch_hadamard_loss = [], []
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            batch_train_loss.append(loss)

            # Compute the Hadamard transform of all possible inputs and add to loss
            landscape = self.model(self.all_inputs).reshape(-1)
            spectrum = torch.matmul(self.H, landscape)
            reg_loss = F.l1_loss(spectrum, torch.zeros_like(spectrum))

            batch_hadamard_loss.append(reg_loss)
            loss += self.config["hadamard_lambda"] * reg_loss

            loss.backward()
            self.optim.step()
        
        return {
            'train_loss': torch.mean(torch.stack(batch_train_loss)).item(),
            'EN_loss': torch.mean(torch.stack(batch_hadamard_loss)).item(),
        }
    
    def ENS_epoch(self):
        device = self.device
        batch_train_loss, batch_hadamard_loss = [], []
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            batch_train_loss.append(loss)

            wht_out = self.model(torch.tensor(self.X_all, dtype=torch.float)).reshape(-1)
            wht_diff = wht_out - torch.tensor(self.Hu, dtype=torch.float) + torch.tensor(self.lam, dtype=torch.float)
            reg_loss = self.config["rho"]/2 * nn.MSELoss(wht_diff,torch.zeros_like(wht_diff))
            loss += self.config["hadamard_lambda"] * reg_loss
            batch_hadamard_loss.append(reg_loss)

            loss.backward()
            self.optim.step()
        
        with torch.no_grad():
            y_hat_all = self.model(torch.from_numpy(self.X_all).float())
        y_hat_all = y_hat_all.numpy().flatten()

        # run spright to do fast sparse WHT
        spright = SPRIGHT('frame', [1,2,3], self.spright_sample)
        spright.set_train_data(self.X_all,  y_hat_all + self.lam, self.X_all_inverse_ind)
        spright.model_to_remove = self.model
        flag = spright.initial_run()
        if flag:
            spright.peel_rest()
            
            M = make_system_simple(np.vstack(spright.model.support), self.X_all)
            self.Hu = np.dot(M,spright.model.coef_)
            
            # update the dual 
            self.lam = self.lam + y_hat_all - self.Hu
        
        return {
            'train_loss': torch.mean(torch.stack(batch_train_loss)).item(),
            'ENS_loss': torch.mean(torch.stack(batch_hadamard_loss)).item(),
        }
