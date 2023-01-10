from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from typing import Dict, List

import glob
import hashlib
import json
import math
import os
import random
import time
import wandb

from epistatic_net.spright_utils import SPRIGHT, make_system_simple
from epistatic_net.wht_sampling import SPRIGHTSample

EVAL_BATCH_SIZE = 204800

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

def hash_dict(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def get_bounded_degree_indices(n, deg_bound):
    """ Returns indices of frequencies with degrees up to the given upper bound 

    Parameters:
        n: int
            - the dimension of the space
        deg_bound:
            - the upper bound for acceptable degrees
    """
    degs = np.sum(list(product([0, 1], repeat=n)), axis=1)
    return list(np.where(degs <= deg_bound)[0])

def signal_error(ground_amps, learned_amps):
    return (np.sum((learned_amps-ground_amps)**2) / np.sum(ground_amps**2)).item()


class HashingLoss:
    def __init__(self, n: int, b: int, deg_bound: int=None, device: str="cuda"):
        """ The  class for computing hashing loss

        Parameters:
            n: int
                - the dimension of the original space
            b: int
                - the dimension of the hashing space
            deg_bound: int (default: None)
                - the upper bound for the degree of frequencies involved in the loss computation
            device: str (default: "cuda")
                - device of the torch tensors
        """
        self.n = n
        self.b = b
        self.deg_bound = deg_bound
        self.device = device

        self.H = hadamard_matrix(b, normalize=False).to(device)
        if deg_bound is not None:
            self.bounded_degree_indices = self.get_bounded_degree_indices()
    
    def get_sample_inputs(self):
        """ Returns input samples sampled by a uniformly random hash matrix """
        hash_sigma = (torch.rand(self.b, self.n) < 0.5).float() # multivariate bernouli with p=0.5
        hash_inputs = torch.asarray(list((product((0.0,1.0), repeat=self.b))))

        sample_inputs = (hash_inputs @ hash_sigma ) % 2
        return sample_inputs

    def get_bounded_degree_indices(self):
        """ Returns indices of frequencies with degrees up to the given upper bound """
        degs = np.sum(list(product([0, 1], repeat=self.b)), axis=1)
        return list(np.where(degs <= self.deg_bound)[0])
    
    def compute_loss(self, model):
        """ Computes L1 norm of approximated (bounded) amplitudes """
        # Find the sample inputs using the hashing scheme
        sample_inputs = self.get_sample_inputs()
        sample_inputs = sample_inputs.to(self.device)

        # Compute the Hadamard transform of sample_inputs
        X = model(sample_inputs)

        ## Limit freqs to bounded degree if requested
        if self.deg_bound is None:
            Y = self.H @ X
        else:
            freq_indices = self.bounded_degree_indices
            Y = self.H[freq_indices] @ X

        return F.l1_loss(Y, torch.zeros_like(Y))


class ModelTrainer:
    def __init__(self, model: nn.Module, train_ds: Dataset, val_ds: Dataset, config: Dict, device = "cuda", log_wandb = False, checkpoint_cache=True, checkpoint_interval=25,
                    report_epoch_fourier=False, print_logs=True, plot_results=False, experiment_name="default", test_ds: Dataset=None, **kwargs):
        '''
            Trains a torch model given the dataset and the training method.

            Args:
                - model: the torch model to train
                - dataset: the torch dataset to use for training
                - training_method: training method to use for training which usually sets how the loss should
                                be calculated in each epoch
                - config: a config dictionary that includes all the hyperparameters for training
                - device: torch device used for the training
                - log_wandb: sets whether log results through wandb
                - report_epoch_fourier: sets if the exact Fourier transform of the model should be computed
                                         and reported.
        '''

        self.device = device
        self.model = model
        self.config = config
        self.log_wandb = log_wandb
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.report_epoch_fourier = report_epoch_fourier
        self.logs = []
        self.plot_results = plot_results
        self.print_logs = print_logs
        self.checkpoint_cache = checkpoint_cache
        self.checkpoint_interval = checkpoint_interval
        self.experiment_name = experiment_name
        self.args = kwargs

        # Set seeds
        random_seed = config["random_seed"]
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Dataloader
        self.train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE)
        self.n = next(iter(self.val_loader))[0].shape[1] # original space dimension

        # Training stuff
        self.optim = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.scheduler = StepLR(self.optim, step_size=config.get("scheduler_step_size", 20), gamma=config.get("scheduler_gamma", 1))

        # Training method
        training_method = config["training_method"]
        if training_method == "normal":
            self.train_epoch = self.normal_epoch

        elif training_method == "EN" or training_method == "bounded_EN":
            self.train_epoch = self.EN_epoch
            self.all_inputs = torch.asarray(list((product((0,1), repeat=self.n)))).float().to(device)
            self.H = hadamard_matrix(self.n, normalize=True).to(device)
            if training_method == "bounded_EN":
                self.H = self.H[get_bounded_degree_indices(self.n, config["deg_bound"])]

        elif training_method == "EN-S" or training_method == "alternate":
            self.train_epoch = self.ENS_epoch
            self.spright_sample = SPRIGHTSample(self.n, config["b"], config["SPRIGHT_d"], random_seed=config["random_seed"])
            self.X_fourier = np.concatenate(
                (
                    np.vstack(self.spright_sample.sampling_locations[0]),
                    np.vstack(self.spright_sample.sampling_locations[1]),
                    np.vstack(self.spright_sample.sampling_locations[2])
                )
            )
            self.X_fourier, self.X_fourier_inverse_ind = np.unique(self.X_fourier, axis=0, return_inverse='True')

            # initialzie ADMM 
            self.Hu = np.zeros(len(self.X_fourier))
            self.lam = np.zeros(len(self.X_fourier))

            if training_method == "alternate":
                self.train_epoch = self.alternate_epoch
                X_train = train_ds[:][0]
                self.X_t = X_train.cpu().numpy()
                self.X_t_tensor = X_train.to(self.device)

                self.spright_support = np.zeros((1,self.n))
                self.spright_coef = np.zeros((1,1))

        elif training_method == "hashing":
            self.train_epoch = self.hashing_epoch
            self.hash_loss = HashingLoss(self.n, config["b"], device=device)

        else:
            raise Exception(f"'{training_method}' training method is not yet implemented.")

        # Fourier of network
        if self.report_epoch_fourier:
            self.original_H = hadamard_matrix(self.n, normalize=True).to(device)

        # Calculating R2
        self.train_size = len(train_ds)
        self.val_size = len(val_ds)
        train_mean_y = sum([torch.sum(y).item() for _, y in self.train_loader]) / len(train_ds)
        val_mean_y = sum([torch.sum(y).item() for _, y in self.val_loader]) / len(val_ds)
        self.train_tss = sum([torch.sum((y-train_mean_y)**2).item() for _, y in self.train_loader])
        self.val_tss = sum([torch.sum((y-val_mean_y)**2).item() for _, y in self.val_loader])
        if self.test_ds is not None:
            self.test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE)
            self.test_size = len(test_ds)
            test_mean_y = sum([torch.sum(y).item() for _, y in self.test_loader]) / len(test_ds)
            self.test_tss = sum([torch.sum((y-test_mean_y)**2).item() for _, y in self.test_loader])

    def train_model(self):
        device = self.device
        self.spectrums = []
        self.model.to(device)
        
        # Start from the latest epoch if available
        start_epoch = 0
        worse_loss_count = 0 # Used to determine early stopping
        if self.checkpoint_cache:
            start_epoch = self.load_from_latest_checkpoint() + 1

        # Compute the Fourier spectrum of Model before training
        if self.report_epoch_fourier:
            if start_epoch == 0:
                self.model.eval()
                with torch.no_grad():
                    all_inputs = torch.asarray(list((product((0.0,1.0), repeat=self.n))))
                    landscape = self.model(all_inputs.to(device))
                    fourier_spectrum = self.original_H @ landscape
                    self.spectrums.append(fourier_spectrum.cpu().numpy())

        for epoch in range(start_epoch, self.config["num_epochs"]):
            # Train epoch
            self.model.train()
            epoch_start = time.time()
            RSS, epoch_log = self.train_epoch()
            epoch_log["train_time"] = time.time() - epoch_start
            epoch_log["train_mse_loss"] = RSS / self.train_size
            epoch_log["train_r2"] = 1 - RSS / self.train_tss

            self.scheduler.step()

            # Evaluate the model on validation set
            epoch_log.update(self.evaluate_epoch())
            
            # Compute the Fourier spectrum of Model if required
            if self.report_epoch_fourier:
                self.spectrums.append(self.compute_fourier_spectrum())

                # Log R2 of learned amps
                learned_amps = self.spectrums[-1]
                selected_freqs = self.args["data_freqs"]
                epoch_log["amp_r2"] = r2_score(self.args["data_spectrum"], learned_amps)
                epoch_log["signal_error"] = signal_error(self.args["data_spectrum"], learned_amps)
                epoch_log["selected_amp_r2"] = r2_score(self.args["data_spectrum"][selected_freqs], learned_amps[selected_freqs])
                epoch_log["selected_signal_error"] = signal_error(self.args["data_spectrum"][selected_freqs], learned_amps[selected_freqs])

            # Compute aggregated metrics
            self.logs.append(epoch_log)
            epoch_log["max_val_r2"] = max([l["val_r2"] for l in self.logs])
            epoch_log["min_val_mse_loss"] = min([l["val_mse_loss"] for l in self.logs])
            if self.test_ds is not None:
                epoch_log["max_test_r2"] = max([l["test_r2"] for l in self.logs])
                epoch_log["min_test_mse_loss"] = min([l["test_mse_loss"] for l in self.logs])
                best_model_ind = np.argmin([l["val_mse_loss"] for l in self.logs])
                epoch_log["best_test_r2"] = self.logs[best_model_ind]["test_r2"]
            if self.report_epoch_fourier:
                epoch_log["max_amp_r2"] = max([l["amp_r2"] for l in self.logs])
                epoch_log["min_signal_error"] = min([l["signal_error"] for l in self.logs])
                epoch_log["max_selected_amp_r2"] = max([l["selected_amp_r2"] for l in self.logs])
                epoch_log["min_selected_signal_error"] = min([l["selected_signal_error"] for l in self.logs])

            # Logging
            if self.print_logs:
                sparsity_loss = ""
                for log_att in epoch_log.keys():
                    if len(log_att.split("_")) == 2 and log_att.split("_")[1] == "loss":
                        sparsity_loss = f"{log_att}: {epoch_log[log_att]:.5f}"

                print(f"#{epoch} - Train Loss: {epoch_log['train_mse_loss']:.3f}, R2: {epoch_log['train_r2']:.3f}"\
                    f"\tValidation Loss: {epoch_log['val_mse_loss']:.3f}, R2: {epoch_log['val_r2']:.3f}\t{sparsity_loss}")
            if self.log_wandb:
                wandb.log(epoch_log)

            # Save checkpoint
            if self.checkpoint_cache:
                # Best model
                if epoch_log["val_mse_loss"] == epoch_log["min_val_mse_loss"]:
                    self.save_model(epoch, best=True)

                # Interval checkpoints
                if (epoch % self.checkpoint_interval == 0) or (epoch == self.config["num_epochs"]-1):
                    self.save_model(epoch)
            
            # Early stop
            best_model_ind = np.argmin([l["val_mse_loss"] for l in self.logs])
            if epoch - best_model_ind >= self.config.get("early_stopping", math.inf):
                print("Halted because of early stopping.")
                self.save_model(epoch)
                break
        
        if self.plot_results:
            self.plot_logs()
            
        if self.report_epoch_fourier:
            return self.spectrums

    def evaluate_model(self, val_loader, val_size, val_tss):
        val_start = time.time()
        self.model.eval()
        with torch.no_grad():
            RSS = 0
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                RSS += torch.sum((y_hat-y)**2).item()
        return {
            "val_mse_loss": RSS / val_size,
            "val_r2": 1 - RSS / val_tss,
            "val_time": time.time() - val_start,
        }
    
    def evaluate_epoch(self):
        val_evaluation = self.evaluate_model(self.val_loader, self.val_size, self.val_tss)
        if self.test_ds is not None:
            test_evaluation = self.evaluate_model(self.test_loader, self.test_size, self.test_tss)
            val_evaluation.update({k.replace("val_","test_"): v for k, v in test_evaluation.items()})
        
        return val_evaluation
    
    def evaluate_model_on_dataset(self, val_ds):
        val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE)
        val_size = len(val_ds)
        val_mean_y = sum([torch.sum(y).item() for _, y in val_loader]) / len(val_ds)
        val_tss = sum([torch.sum((y-val_mean_y)**2).item() for _, y in val_loader])
        return self.evaluate_model(val_loader, val_size, val_tss)
    
    def compute_fourier_spectrum(self):
        self.model.eval()
        with torch.no_grad():
            all_inputs = torch.asarray(list((product((0.0,1.0), repeat=self.n))))
            landscape = self.model(all_inputs.to(self.device))
            fourier_spectrum = self.original_H @ landscape

        return fourier_spectrum.cpu().numpy()
    
    def plot_logs(self):
        loss_data = {"train_mse_loss": [l["train_mse_loss"] for l in self.logs],
                    "val_mse_loss": [l["val_mse_loss"] for l in self.logs]}
        r2_data = {"train_r2": [l["train_r2"] for l in self.logs],
                    "val_r2": [l["val_r2"] for l in self.logs]}

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
        fig.suptitle(self.config["training_method"])

        sns.lineplot(data=loss_data, ax=axes[0])
        sns.lineplot(data=r2_data, ax=axes[1])
        axes[1].set_ylim(0, 1)

    def get_model_dir(self):
        data_directory = os.environ.get("EXPERIMENT_DATA") if "EXPERIMENT_DATA" in os.environ else os.getcwd()

        keys_to_ignore = ["num_epochs"]
        config_to_hash = {k:v for k, v in self.config.items() if k not in keys_to_ignore}
        config_hash = hash_dict(config_to_hash)

        model_dir = f"{data_directory}/checkpoints/{self.experiment_name}/{config_hash}/"
        # Create directory and write config if not existed before
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            with open(f'{model_dir}/config.json', 'w') as fp:
                json.dump(config_to_hash, fp, sort_keys=True)
        
        return model_dir
    
    def save_model(self, epoch, best=False):
        # Save torch stuff
        model_dir = self.get_model_dir()
        checkpoint = { 
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }
        
        if best:
            torch.save(checkpoint, f'{model_dir}/best.pth')
        else:
            torch.save(checkpoint, f'{model_dir}/{epoch}.pth')

            # Save logs (+remove older ones)
            for f in glob.glob(f'{model_dir}/log*.json'):
                os.remove(f)
            with open(f'{model_dir}/log{epoch}.json', 'w') as fp:
                json.dump(self.logs, fp)

            # Save Fourier spectrums (+remove older ones)
            if self.report_epoch_fourier:
                for f in glob.glob(f'{model_dir}/spectrums*.npy'):
                    os.remove(f)
                np.save(f'{model_dir}/spectrums{epoch}.npy', self.spectrums, allow_pickle=True)
        

    def load_model(self, epoch):
        model_dir = self.get_model_dir()
        # Load torch stuff
        checkpoint_file = f'{model_dir}/{epoch}.pth'
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint["model"])
            self.optim.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            raise Exception(f"Could not find {checkpoint_file} to load from.")
        
        # Load logs
        with open(f'{model_dir}/log{epoch}.json') as log_file:
            self.logs = json.load(log_file)
            # Re-report previous logs by wandb
            if self.log_wandb:
                for log in self.logs:
                    wandb.log(log)
        
        # Load Fourier Spectrums
        spectrum_file = f'{model_dir}/spectrums{epoch}.npy'
        if os.path.exists(spectrum_file):
            self.spectrums = list(np.load(spectrum_file, allow_pickle=True))

        print(f"Model state is loaded from checkpoint of eopch {epoch}. Continuing from this checkpoint.")
    
    def load_from_latest_checkpoint(self):
        model_dir = self.get_model_dir()
        available_epochs = [int(os.path.basename(f).split(".")[0]) for f in glob.glob(f"{model_dir}/*.pth") 
                            if os.path.basename(f) != 'best.pth']
        if len(available_epochs) > 0:
            latest_epoch = max(available_epochs)
            self.load_model(latest_epoch)
        else:
            latest_epoch = -1

        return latest_epoch
    
    def normal_epoch(self):
        device = self.device
        RSS = 0
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
        batch_hadamard_loss = []
        hadamard_times = []
        RSS = 0
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            RSS += torch.sum((y_hat-y)**2).item()

            # Compute the Hadamard transform of sample_inputs and add to loss
            hadamard_start = time.time()

            hadamard_loss = 0
            hash_repetition = self.config.get("hash_repetition", 1)
            for i in range(hash_repetition):
                hadamard_loss += self.hash_loss.compute_loss(self.model)
            loss += self.config["hadamard_lambda"] * hadamard_loss / hash_repetition

            hadamard_times.append(time.time() - hadamard_start)
            batch_hadamard_loss.append(hadamard_loss.item() / hash_repetition)

            loss.backward()
            self.optim.step()
        
        return RSS, {
            'hadamard_time': sum(hadamard_times),
            'hadamard_iteration_time': np.mean(hadamard_times),
            'hashing_loss': np.mean(batch_hadamard_loss),
        }
    
    def EN_epoch(self):
        device = self.device
        batch_hadamard_loss = []
        hadamard_times = []
        RSS = 0
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
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
        batch_hadamard_loss = []
        RSS = 0
        network_update_time = time.time()
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            RSS += torch.sum((y_hat-y)**2).item()

            wht_out = self.model(torch.tensor(self.X_fourier, dtype=torch.float, device=device)).reshape(-1)
            wht_diff = wht_out - torch.tensor(self.Hu, dtype=torch.float, device=device) + torch.tensor(self.lam, dtype=torch.float, device=device)
            reg_loss = self.config["rho"]/2 * l2_loss(wht_diff,torch.zeros_like(wht_diff))
            loss += self.config["hadamard_lambda"] * reg_loss
            batch_hadamard_loss.append(reg_loss.item())

            loss.backward()
            self.optim.step()
        network_update_time = time.time() - network_update_time
        
        hadamard_start = time.time()

        with torch.no_grad():
            y_hat_fourier = self.model(torch.tensor(self.X_fourier, dtype=torch.float, device=device))
            y_hat_fourier = y_hat_fourier.cpu().numpy().flatten()

        # run spright to do fast sparse WHT
        fourier_start = time.time()
        spright = SPRIGHT('frame', [1,2,3], self.spright_sample)
        spright.set_train_data(self.X_fourier,  y_hat_fourier + self.lam, self.X_fourier_inverse_ind)
        # spright.model_to_remove = self.model
        flag = spright.initial_run()
        if flag:
            spright.peel_rest()
            
            M = make_system_simple(np.vstack(spright.model.support), self.X_fourier)
            self.Hu = np.dot(M,spright.model.coef_)
            
            # update the dual 
            self.lam = self.lam + y_hat_fourier - self.Hu
        
        return RSS, {
            'network_update_time': network_update_time,
            'hadamard_time': time.time() - hadamard_start,
            'fourier_time': time.time() - fourier_start,
            'ENS_loss': np.mean(batch_hadamard_loss),
        }

    def alternate_epoch(self):
        device = self.device
        l2_loss = nn.MSELoss()
        batch_hadamard_loss = []
        RSS = 0
        network_update_time = time.time()
        for X, y in self.train_loader:
            self.optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X)
            loss = F.mse_loss(y, y_hat)
            RSS += torch.sum((y_hat-y)**2).item()

            # Fourier loss
            if self.config.get("intense_regularization", False):
                y_t = self.model(self.X_t_tensor)
                M = make_system_simple(self.spright_support, self.X_t)
                y_fourier = np.dot(M,self.spright_coef).flatten()
                reg_loss = F.mse_loss(y_t, torch.tensor(y_fourier, dtype=torch.float, device=device))
            else:
                M = make_system_simple(self.spright_support, X.cpu().numpy())
                y_fourier = np.dot(M,self.spright_coef).flatten()
                reg_loss = F.mse_loss(y_hat, torch.tensor(y_fourier, dtype=torch.float, device=device))
            
            loss += self.config["hadamard_lambda"] * reg_loss
            batch_hadamard_loss.append(reg_loss.item())

            loss.backward()
            self.optim.step()
        network_update_time = time.time() - network_update_time
        
        hadamard_start = time.time()

        with torch.no_grad():
            y_hat_fourier = self.model(torch.tensor(self.X_fourier, dtype=torch.float, device=device))
            y_hat_fourier = y_hat_fourier.cpu().numpy().flatten()

        # run spright to do fast sparse WHT
        fourier_start = time.time()
        spright = SPRIGHT('frame', [1,2,3], self.spright_sample)
        spright.set_train_data(self.X_fourier,  y_hat_fourier, self.X_fourier_inverse_ind)

        flag = spright.initial_run()
        if flag:
            spright.peel_rest()
            self.spright_support = spright.model.support
            self.spright_coef = spright.model.coef_

        fourier_time = time.time() - fourier_start
        hadamard_time = time.time() - hadamard_start
        
        return RSS, {
            'network_update_time': network_update_time,
            'hadamard_time': hadamard_time,
            'fourier_time': fourier_time,
            'alternate_loss': np.mean(batch_hadamard_loss),
        }