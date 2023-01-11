import pickle
import numpy as np
from sample_optimal_sparse_hadamard.sparseWHT_robust_sampleoptimal import SWHTRobust
import sample_optimal_sparse_hadamard.utils.parallelized_sampling_interface as parallelized_sampling_interface
# import models.nn_model
from sklearn.metrics import r2_score, mean_squared_error, max_error
import pathlib
import torch
from torch.utils.data import DataLoader, Dataset
import utils
from sample_optimal_sparse_hadamard.fourier import Fourier
import os

# Global settings
# Fourier transform drops any coefficient with absolute values smaller than espilon
EPSILON = 0.00001
# Test set size to training set size ratio
TEST_SET_RATIO = 0.2
# Random seed used in train test split
RANDOM_SEED = 0


def get_fourier_time_samples_with_dummy_function(n ,k, degree, random_seed=0, cache=True):
    """ Get time samples required by Fourier transform by passing in a dummy function
    """
    data_directory = os.environ.get("EXPERIMENT_DATA") if "EXPERIMENT_DATA" in os.environ else os.getcwd()
    file_dir = f"{data_directory}/swht"
    file_name = f"n={n}_k={k}_degree={degree}_seed{random_seed}.pkl"
    file_path = f"{file_dir}/{file_name}"

    if cache and os.path.exists(file_path):
        return np.load(file_path)

    sparse_wht = SWHTRobust(n, k, C=1, finite_field_class="reed_solomon_cs", degree=degree, robust_iterations=1)
    f = parallelized_sampling_interface.DummyFunction()
    sparse_wht.run(f, random_seed=random_seed)
    time_samples = np.vstack(f.get_time_samples())
    
    if cache:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        np.save(file_path, time_samples)
    
    return time_samples


def compute_fourier(n, k, d, predictions, random_seed=0, finite_field_class="reed_solomon_cs", 
                    C=1, robust_iterations=1, epsilon=0.00001):
    f = parallelized_sampling_interface.FastFunction(pred_array=predictions)
    sparse_wht = SWHTRobust(n=n, K=k, C=C, finite_field_class=finite_field_class, degree=d,
                            robust_iterations=robust_iterations, epsilon=epsilon)
    fourier_transform = sparse_wht.run(f, random_seed=random_seed)

    return fourier_transform

if __name__ == "__main__":
    print(get_fourier_time_samples_with_dummy_function(10, 5, 2))
