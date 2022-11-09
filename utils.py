import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import StepLR
from typing import Dict, List

from itertools import product
import math
import random
import time
import wandb

from epistatic_net.spright_utils import SPRIGHT, make_system_simple
from epistatic_net.wht_sampling import SPRIGHTSample

class FourierDataset(Dataset):
    def __init__(self, n, k, freq_sampling_method="uniform_deg", amp_sampling_method="random", d=None, p_freq=None, n_samples = 100, p_t=0.5, scale_y=True):
        self.n = n
        self.k = k
        self.scale_y = scale_y

        if freq_sampling_method == "uniform_deg":
            self.freq_f = self.uniform_deg_freq(d)
        elif freq_sampling_method == "bernouli":
            self.freq_f = self.bernouli_freq(p_freq)
        elif freq_sampling_method == "bounded_deg":
            self.freq_f = self.bounded_degree_freq(d)
        else:
            raise Exception(f'"{freq_sampling_method}" is not a generation method for FourierDataset.')

        if amp_sampling_method == "random":
            self.amp_f = torch.FloatTensor(k).uniform_(-1, 1)
        elif amp_sampling_method == "constant":
            self.amp_f = torch.ones(k)

        self.X = (torch.rand(n_samples, n) < p_t).float()
        self.y = self.compute_y(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def compute_y(self, X):
        t_dot_f = X @ torch.t(self.freq_f)
        return torch.sum(torch.where(t_dot_f % 2 == 1, -1, 1) * self.amp_f, axis = -1) / (self.k if self.scale_y else (2**self.n))
    
    def bounded_degree_freq(self, d):
        freqs = torch.zeros(self.k, self.n)
        one_indices = [(i, random.randrange(self.n)) for j in range(d) for i in range(self.k)]
        freqs[list(zip(*one_indices))] = 1.0
        return freqs

    def bernouli_freq(self, p):
        return (torch.rand(self.k, self.n) < p).float()

    def uniform_deg_freq(self, d):
        freqs = []
        for i in range(self.k):
            deg = random.randint(1, d)
            is_duplicate = True
            while is_duplicate:
                one_indices = random.sample(range(self.n), deg)
                new_f = [1.0 if i in one_indices else 0.0 for i in range(self.n)]
                if new_f not in freqs:
                    freqs.append(new_f)
                    is_duplicate = False
        return torch.tensor(freqs)

    def get_int_freqs(self):
        return self.freq_f.cpu().numpy().dot(2**np.arange(self.n)[::-1]).astype(int)

    def get_fourier_spectrum(self):
        spectrum = np.zeros(2**self.n)
        spectrum[self.get_int_freqs()] = self.amp_f.cpu().numpy()
        return spectrum


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

    sample_inputs = (hash_inputs @ hash_sigma ) % 2
    return sample_inputs


class ModelTrainer:
    def __init__(self, model: nn.Module, dataset: Dataset, config: Dict, p_val = 0.25, device = "cuda", 
                    log_wandb = False, report_epoch_fourier=False, print_logs=True, plot_results=False):
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
                - log_wandb: sets whether log results through wandb
                - report_epoch_fourier: sets if the exact Fourier transform of the model should be computed
                                         and reported.
        '''

        self.device = device
        self.model = model
        self.config = config
        self.log_wandb = log_wandb
        self.dataset = dataset
        self.report_epoch_fourier = report_epoch_fourier
        self.logs = []
        self.plot_results = plot_results
        self.print_logs = print_logs

        # Dataloader
        self.val_size = int(p_val * len(self.dataset))
        self.train_size = len(self.dataset) - self.val_size
        train_ds, val_ds = random_split(self.dataset, lengths=[self.train_size, self.val_size])

        self.train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=True)
        self.n = next(iter(self.val_loader))[0].shape[1] # original space dimension

        # Training method
        training_method = config["training_method"]
        if training_method == "normal":
            self.train_epoch = self.normal_epoch

        elif training_method == "EN":
            self.train_epoch = self.EN_epoch
            self.all_inputs = torch.asarray(list((product((0,1), repeat=self.n)))).to(device)
            self.H = hadamard_matrix(self.n, normalize=True).to(device)

        elif training_method == "EN-S":
            self.train_epoch = self.ENS_epoch
            self.spright_sample = SPRIGHTSample(self.n, config["b"], config["SPRIGHT_d"], random_seed=config["random_seed"])
            self.X_all = np.concatenate(
                (
                    np.vstack(self.spright_sample.sampling_locations[0]),
                    np.vstack(self.spright_sample.sampling_locations[1]),
                    np.vstack(self.spright_sample.sampling_locations[2])
                )
            )
            self.X_all, self.X_all_inverse_ind = np.unique(self.X_all, axis=0, return_inverse='True')
            X_all_ds = TensorDataset(torch.from_numpy(self.X_all).to("cpu"), torch.zeros(self.X_all.shape[0], device="cpu"))
            self.X_all_loader = DataLoader(X_all_ds, batch_size=10240)

            # initialzie ADMM 
            self.Hu = np.zeros(len(self.X_all))
            self.lam = np.zeros(len(self.X_all))

        elif training_method == "hashing":
            self.train_epoch = self.hashing_epoch
            self.H = hadamard_matrix(config["b"], normalize=False).to(device)
            
        else:
            raise Exception(f"'{training_method}' training method is not yet implemented.")

        # Training stuff
        self.optim = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.scheduler = StepLR(self.optim, step_size=config.get("scheduler_step_size", 20), gamma=config.get("scheduler_gamma", 1))

        # Fourier of network
        if self.report_epoch_fourier:
            self.original_H = hadamard_matrix(self.n, normalize=True).to(device)

        # Calculating R2
        train_mean_y = sum([torch.sum(y).item() for _, y in self.train_loader]) / len(train_ds)
        val_mean_y = sum([torch.sum(y).item() for _, y in self.val_loader]) / len(val_ds)
        self.train_tss = sum([torch.sum((y-train_mean_y)**2).item() for _, y in self.train_loader])
        self.val_tss = sum([torch.sum((y-val_mean_y)**2).item() for _, y in self.val_loader])

    def train_model(self):
        device = self.device
        spectrums = []
        self.model.to(device)

        # Compute the Fourier spectrum of Model before training
        if self.report_epoch_fourier:
            self.model.eval()
            with torch.no_grad():
                all_inputs = torch.asarray(list((product((0.0,1.0), repeat=self.n))))
                landscape = self.model(all_inputs.to(device))
                fourier_spectrum = self.original_H @ landscape
                spectrums.append(fourier_spectrum.cpu().numpy())

        for epoch in range(self.config["num_epochs"]):
            # Train epoch based on the set training method
            self.model.train()
            epoch_start = time.time()
            RSS, epoch_log = self.train_epoch()
            epoch_log["train_time"] = time.time() - epoch_start
            epoch_log["train_mse_loss"] = RSS / self.train_size
            epoch_log["train_r2"] = 1 - RSS / self.train_tss

            # Evaluate the model on validation set
            val_start = time.time()
            self.model.eval()
            with torch.no_grad():
                RSS = 0
                for X, y in self.val_loader:
                    X, y = X.to(device), y.to(device)
                    y_hat = self.model(X)
                    RSS += torch.sum((y_hat-y)**2).item()
                
                # Compute the Fourier spectrum of Model if required
                if self.report_epoch_fourier:
                    all_inputs = torch.asarray(list((product((0.0,1.0), repeat=self.n))))
                    landscape = self.model(all_inputs.to(device))
                    fourier_spectrum = self.original_H @ landscape
                    spectrums.append(fourier_spectrum.cpu().numpy())

                    # Log R2 of learned amps
                    learned_amps = spectrums[-1][self.dataset.get_int_freqs()]
                    epoch_log["amp_r2"] = r2_score(self.dataset.amp_f.cpu().numpy(), learned_amps)
            
            epoch_log["val_mse_loss"] = RSS / self.val_size
            epoch_log["val_r2"] = 1 - RSS / self.val_tss
            epoch_log["val_time"] = time.time() - val_start
            if self.print_logs:
                print(f"#{epoch} - Train Loss: {epoch_log['train_mse_loss']:.3f}, R2: {epoch_log['train_r2']:.3f}"\
                    f"\tValidation Loss: {epoch_log['val_mse_loss']:.3f}, R2: {epoch_log['val_r2']:.3f}")

            # Log wandb
            self.logs.append(epoch_log)
            epoch_log["max_val_r2"] = max([l["val_r2"] for l in self.logs])
            epoch_log["min_val_mse_loss"] = min([l["val_mse_loss"] for l in self.logs])
            if self.report_epoch_fourier:
                epoch_log["max_amp_r2"] = max([l["amp_r2"] for l in self.logs])

            if self.log_wandb:
                wandb.log(epoch_log)
            
            self.scheduler.step()
        
        if self.plot_results:
            self.plot_logs()
            
        if self.report_epoch_fourier:
            return spectrums
    
    def plot_logs(self):
        loss_data = {"train_mse_loss": [l["train_mse_loss"] for l in self.logs],
                    "val_mse_loss": [l["val_mse_loss"] for l in self.logs]}
        r2_data = {"train_r2": [l["train_r2"] for l in self.logs],
                    "val_r2": [l["val_r2"] for l in self.logs]}

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
        fig.suptitle(self.config["training_method"])

        sns.lineplot(loss_data, ax=axes[0])
        sns.lineplot(r2_data, ax=axes[1])
        axes[1].set_ylim(0, 1)
    
    def normal_epoch(self):
        device = self.device
        RSS = 0
        batch_train_loss = []
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            RSS += torch.sum((y_hat-y)**2).item()

            loss.backward()
            self.optim.step()
        
        return RSS, {}
    
    def hashing_epoch(self):
        device = self.device
        batch_train_loss, batch_hadamard_loss = [], []
        hadamard_times = []
        RSS = 0
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            batch_train_loss.append(loss)
            RSS += torch.sum((y_hat-y)**2).item()

            # Find the sample inputs using the hashing scheme
            sample_inputs = get_sample_inputs(self.n, self.config["b"])
            sample_inputs = sample_inputs.to(device)

            # Compute the Hadamard transform of sample_inputs and add to loss
            hadamard_start = time.time()
            X = self.model(sample_inputs)
            Y = self.H @ X
            hadamard_loss = F.l1_loss(Y, torch.zeros_like(Y))

            hadamard_times.append(time.time() - hadamard_start)
            batch_hadamard_loss.append(hadamard_loss)
            loss += self.config["hadamard_lambda"] * hadamard_loss

            loss.backward()
            self.optim.step()
        
        return RSS, {
            'hadamard_time': sum(hadamard_times),
            'hadamard_iteration_time': np.mean(hadamard_times),
            'hashing_loss': torch.mean(torch.stack(batch_hadamard_loss)).item(),
        }
    
    def EN_epoch(self):
        device = self.device
        batch_train_loss, batch_hadamard_loss = [], []
        hadamard_times = []
        RSS = 0
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            batch_train_loss.append(loss)
            RSS += torch.sum((y_hat-y)**2).item()

            # Compute the Hadamard transform of all possible inputs and add to loss
            hadamard_start = time.time()
            landscape = self.model(self.all_inputs).reshape(-1)
            spectrum = torch.matmul(self.H, landscape)
            reg_loss = F.l1_loss(spectrum, torch.zeros_like(spectrum))
            hadamard_times.append(time.time() - hadamard_start)

            batch_hadamard_loss.append(reg_loss)
            loss += self.config["hadamard_lambda"] * reg_loss

            loss.backward()
            self.optim.step()
        
        return RSS, {
            'hadamard_time': sum(hadamard_times),
            'hadamard_iteration_time': np.mean(hadamard_times),
            'EN_loss': torch.mean(torch.stack(batch_hadamard_loss)).item(),
        }
    
    def ENS_epoch(self):
        device = self.device
        l2_loss = nn.MSELoss()
        batch_train_loss, batch_hadamard_loss = [], []
        RSS = 0
        network_update_time = time.time()
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            batch_train_loss.append(loss)
            RSS += torch.sum((y_hat-y)**2).item()

            wht_out = self.model(torch.tensor(self.X_all, dtype=torch.float, device=device)).reshape(-1)
            wht_diff = wht_out - torch.tensor(self.Hu, dtype=torch.float, device=device) + torch.tensor(self.lam, dtype=torch.float, device=device)
            reg_loss = self.config["rho"]/2 * l2_loss(wht_diff,torch.zeros_like(wht_diff))
            loss += self.config["hadamard_lambda"] * reg_loss
            batch_hadamard_loss.append(reg_loss)

            loss.backward()
            self.optim.step()
        network_update_time = time.time() - network_update_time
        
        hadamard_start = time.time()

        with torch.no_grad():
            y_hat_alls = []
            for X, _ in self.X_all_loader:
                y_hat_all = self.model(X.float().to(device))
                y_hat_alls.append(y_hat_all.cpu().numpy().flatten())
        y_hat_all = np.concatenate(y_hat_alls)

        # run spright to do fast sparse WHT
        fourier_start = time.time()
        spright = SPRIGHT('frame', [1,2,3], self.spright_sample)
        spright.set_train_data(self.X_all,  y_hat_all + self.lam, self.X_all_inverse_ind)
        # spright.model_to_remove = self.model
        flag = spright.initial_run()
        if flag:
            spright.peel_rest()
            
            M = make_system_simple(np.vstack(spright.model.support), self.X_all)
            self.Hu = np.dot(M,spright.model.coef_)
            
            # update the dual 
            self.lam = self.lam + y_hat_all - self.Hu
        
        return RSS, {
            'network_update_time': network_update_time,
            'hadamard_time': time.time() - hadamard_start,
            'fourier_time': time.time() - fourier_start,
            'ENS_loss': torch.mean(torch.stack(batch_hadamard_loss)).item(),
        }
