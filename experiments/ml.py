import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import wandb

from datasets import get_real_dataset

def main():  
    run = wandb.init()
    print(wandb.config)
    config = wandb.config

    # Dataset
    dataset = get_real_dataset(config["dataset"])
    if config.get("normalize_data", False):
        dataset.y = (dataset.y-torch.mean(dataset.y))/torch.std(dataset.y)
    
    torch.manual_seed(config["fix_seed"])
    config["val_size"] = config.get("val_size", (len(dataset) - config["train_size"])//2)
    remainder = len(dataset) - config["train_size"] - config["val_size"] * 2
    train_ds, val_ds, test_ds, _ = torch.utils.data.random_split(dataset, [config["train_size"], config["val_size"], config["val_size"], remainder])

    # Train model
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    if config["training_method"] == "Lasso":
        model = Lasso(alpha=config["lasso_alpha"], random_state=config["random_seed"])
    elif config["training_method"] == "Random Forest":
        model =  RandomForestRegressor(n_estimators=config["rf_n_estimators"], max_depth=config["rf_max_depth"], random_state=config["random_seed"])
    elif config["training_method"] == "XGBoost":
        model = XGBRegressor(n_estimators=config["rf_n_estimators"], max_depth=config["rf_max_depth"], random_state=config["random_seed"])
    else:
        raise Exception(f"Method {config['training_method']} is not supported.")

    model.fit(train_ds[:][0].cpu().numpy(), train_ds[:][1].cpu().numpy())

    train_pred = model.predict(train_ds[:][0].cpu().numpy())
    val_pred = model.predict(val_ds[:][0].cpu().numpy())
    test_pred = model.predict(test_ds[:][0].cpu().numpy())

    # Log results on wandb
    log = {
            "train_mse_loss": mean_squared_error(train_ds[:][1].cpu().numpy(), train_pred),
            "train_r2": r2_score(train_ds[:][1].cpu().numpy(), train_pred),
            "val_mse_loss": mean_squared_error(val_ds[:][1].cpu().numpy(), val_pred),
            "val_r2": r2_score(val_ds[:][1].cpu().numpy(), val_pred),
            "test_mse_loss": mean_squared_error(test_ds[:][1].cpu().numpy(), test_pred),
            "test_r2": r2_score(test_ds[:][1].cpu().numpy(), test_pred),
        }
    log["min_train_mse_loss"] = log["train_mse_loss"]
    log["min_val_mse_loss"] = log["val_mse_loss"]
    log["min_test_mse_loss"] = log["test_mse_loss"]
    log["max_train_r2"] = log["train_r2"]
    log["max_val_r2"] = log["val_r2"]
    log["max_test_r2"] = log["test_r2"]
    log["best_test_r2"] = log["test_r2"]

    wandb.log(log)

if __name__ == "__main__":
    main()
