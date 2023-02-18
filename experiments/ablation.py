import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from itertools import product
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import wandb
from tqdm.auto import tqdm

from datasets import get_real_dataset

from rf_fourier.fourier import Fourier
from rf_fourier.rf_fourier_extractor import RFFourierExtractor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_directory = os.environ.get("EXPERIMENT_DATA") if "EXPERIMENT_DATA" in os.environ else os.getcwd()

def main():  
    run = wandb.init()
    print(wandb.config)
    config = wandb.config
    print(device)

    random_seed = config.random_seed
    dataset_name = config.dataset
    train_size = config.train_size
    val_size = config.val_size
    rf_n_estimators = config.rf_n_estimators
    rf_max_depth = config.rf_max_depth

    model_name = f"{dataset_name}_train{train_size}_nest{rf_n_estimators}_depth{rf_max_depth}_seed{random_seed}"
    fourier_file = f"{data_directory}/rf_fourier/{model_name}.pkl"
    result_file = f"{data_directory}/rf_fourier/result_{model_name}.pkl"
    rf_file = f"{data_directory}/rf_model/{model_name}.pkl"
    plot_file = f"{data_directory}/plots/ablation_pruned/{model_name}.pdf"

    dataset = get_real_dataset(dataset_name)
    dataset.y = (dataset.y-torch.mean(dataset.y))/torch.std(dataset.y)
    total_size = len(dataset)
    torch.manual_seed(random_seed)
    train_ds, val_ds, _ = torch.utils.data.random_split(dataset, [train_size, val_size, total_size - val_size - train_size])

    # test_pred = model.predict(test_ds[:][0].cpu().numpy())
    y_train = train_ds[:][1].cpu().numpy()
    y_val = val_ds[:][1].cpu().numpy()

    if not os.path.exists(rf_file):
        model =  RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=random_seed)
        model.fit(train_ds[:][0].cpu().numpy(), y_train)
        with open(rf_file, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(rf_file, 'rb') as handle:
            model = pickle.load(handle)
    
    train_pred = model.predict(train_ds[:][0].cpu().numpy())
    val_pred = model.predict(val_ds[:][0].cpu().numpy())

    log = {
            "train_mse_loss": mean_squared_error(y_train, train_pred),
            "train_r2": r2_score(y_train, train_pred),
            "val_mse_loss": mean_squared_error(y_val, val_pred),
            "val_r2": r2_score(y_val, val_pred),
            # "test_mse_loss": mean_squared_error(test_ds[:][1].cpu().numpy(), test_pred),
            # "test_r2": r2_score(test_ds[:][1].cpu().numpy(), test_pred),
        }
    
    if not os.path.exists(fourier_file):
        rf_fourier = RFFourierExtractor(model).get_fourier_transform()

        with open(fourier_file, 'wb') as handle:
            pickle.dump(rf_fourier, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(fourier_file, 'rb') as handle:
            rf_fourier = pickle.load(handle)
        print("Sparsity =", len(rf_fourier.series))
    
    rf_fourier = Fourier(rf_fourier.series, tolerance=1e-4)
    print("New sparsity:", len(rf_fourier.series))
    
    val_torch = val_ds[:][0].clone()
    val_torch.to(device)
    train_torch = train_ds[:][0].clone()
    train_torch.to(device)
    train_fourier_preds = rf_fourier.predict_torch(train_torch)
    val_fourier_preds = rf_fourier.predict_torch(val_torch)

    fourier_log = {
            "fourier_train_r2": r2_score(y_train, train_fourier_preds),
            "fourier_val_r2": r2_score(y_val, val_fourier_preds),
            "train_fourier_r2": r2_score(train_pred, train_fourier_preds),
            "val_fourier_r2": r2_score(val_pred, val_fourier_preds),
        }
    log.update(fourier_log)
    wandb.log(log)

    # Experiment and plots
    fourier_seris = rf_fourier.series

    deg_list = np.array([(len(f), f) for f in fourier_seris.keys()])
    amp_list = np.array(list(fourier_seris.values()))
    argsort_freq = np.argsort(np.abs(amp_list))
    argsort_deg = np.argsort([(f_t[0], random.random()) for f_t in deg_list], axis=0)[:,0]


    if not os.path.exists(result_file):
        results = [{
            "Percentage of removed frequencies": 0,
            "Higher degrees removed": fourier_log["fourier_val_r2"],
            "Lower amplitudes removed": fourier_log["fourier_val_r2"],
        }]

        for deletion_percentage in tqdm(range(5, 100, 5), total=20):
            num_deletions = math.floor(len(deg_list) * deletion_percentage / 100)

            # Remove high degrees
            mask = argsort_deg[:-num_deletions]
            new_freqs = [f_t[1] for f_t in deg_list[mask]]
            new_amps = amp_list[mask]
            new_fourier_deg = Fourier(dict(zip(new_freqs, new_amps)))

            # Remove low amplitudes
            mask = argsort_freq[num_deletions:]
            new_freqs = [f_t[1] for f_t in deg_list[mask]]
            new_amps = amp_list[mask]
            new_fourier_amp = Fourier(dict(zip(new_freqs, new_amps)))

            deletion_log = {
                "Percentage of removed frequencies": deletion_percentage,
                "Higher degrees removed": r2_score(y_val, new_fourier_deg.predict_torch(val_torch)),
                "Lower amplitudes removed": r2_score(y_val, new_fourier_amp.predict_torch(val_torch)),
            }
            
            results.append(deletion_log)
        
        with open(result_file, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(result_file, 'rb') as handle:
            results = pickle.load(handle)
        
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    sns.lineplot(pd.DataFrame.from_dict(results).set_index("Percentage of removed frequencies"), ax=ax)
    ax.set_ylabel("Test $R^2$")
    ax.set_ylim((0, 1))
    ax.legend(loc='lower left').set_zorder(102)
    fig.tight_layout()
    plt.savefig(plot_file)

if __name__ == "__main__":
    main()
