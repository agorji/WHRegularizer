import pickle
import numpy as np
from sparseWHT_robust_sampleoptimal import SWHTRobust
import utils.parallelized_sampling_interface as parallelized_sampling_interface
import models.nn_model
from sklearn.metrics import r2_score, mean_squared_error, max_error
import pathlib
import torch
from torch.utils.data import DataLoader, Dataset
import utils
from utils import EPSILON, TEST_SET_RATIO, RANDOM_SEED
from fourier import Fourier


def get_fourier_from_cache(task_name, k, degree):
    """Load Fourier transform from cache
    """
    this_directory = pathlib.Path(__file__).parent.resolve()
    try:
        with open(f'{this_directory}/cache/{task_name}_k={k}_degree={degree}.pkl', 'rb') as f:
            fourier_transform = pickle.load(f)
            return fourier_transform
    except IOError:
        print("Not found in cache!")


def get_fourier_time_samples_with_dummy_function(task_name, k, degree, export_to_file=True):
    """ Get time samples required by Fourier transform by passing in a dummy function
    """
    dataset_no_features = utils.get_dataset_settings()["no_features"]
    n_var = dataset_no_features[task_name]
    sparse_wht = SWHTRobust(n_var, k, C=1, finite_field_class="reed_solomon_cs", degree=degree,
                            robust_iterations=1)
    f = parallelized_sampling_interface.DummyFunction()
    sparse_wht.run(f)

    this_directory = pathlib.Path(__file__).parent.resolve()
    if export_to_file:
        f.export_times_samples_to_file(dir=f"{this_directory}/time_samples", filename=f"{task_name}_k={k}_degree={degree}.pkl")
    time_samples = np.vstack(f.get_time_samples())
    return time_samples

class NNFourierDataset(Dataset):
    def __init__(self, time_samples):
        self.no_samples, self.no_features = time_samples.shape
        self.time_samples = np.array(time_samples, dtype=np.float64)
        pass
    def __len__(self):
        return self.no_samples

    def __getitem__(self, idx):
        return self.time_samples[idx]


def get_neural_net_predictions(task_name, time_samples, save_to_cache=True, time_samples_config=None):
    """ Compute neural network predictions evaluated on the time samples
    """
    model = models.nn_model.load_model(task_name, best=True)
    dataset = NNFourierDataset(time_samples)
    dataset_batch_size = utils.get_dataset_settings()["batch_size"]
    dataloader = DataLoader(dataset, batch_size=int(dataset_batch_size[task_name]), shuffle=False)
    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.cuda()
            yhat = model(inputs).squeeze()
            yhat = yhat.cpu().numpy()
            predictions.append(yhat)
    predictions = np.concatenate(predictions, axis=0)
    if save_to_cache:
        try:
            this_directory = pathlib.Path(__file__).parent.resolve()
            path_to_value_samples = f"{this_directory}/value_samples/{task_name}_k=" \
                                    f"{time_samples_config['k']}_degree={time_samples_config['degree']}.pkl"
            with open(path_to_value_samples, "wb") as f:
                pickle.dump(predictions, f)
        except KeyError:
            print("In order to save neural net predictions to a file you need to pass in a dict containing k and "
                  "degree to this function so a name can be derived for the saved file")

    return predictions


def compute_fourier(task_name, predictions, save_to_cache=True, time_samples_config=None):
    f = parallelized_sampling_interface.FastFunction(pred_array=predictions)
    dataset_no_features = utils.get_dataset_settings()["no_features"]
    sparse_wht = SWHTRobust(n=dataset_no_features[task_name], K=time_samples_config["k"], C=1,
                            finite_field_class="reed_solomon_cs", degree=time_samples_config["degree"],
                            robust_iterations=1, epsilon=EPSILON)
    fourier_transform = sparse_wht.run(f)
    if save_to_cache:
        this_directory = pathlib.Path(__file__).parent.resolve()
        path_to_cache = f"{this_directory}/cache/{task_name}_k={time_samples_config['k']}_" \
                        f"degree={time_samples_config['degree']}.pkl"
        with open(path_to_cache, "wb") as file:
            pickle.dump(fourier_transform, file)

    return fourier_transform


def test_fourier_quality(task_name, fourier_transform):
    x_train, x_test, y_train, y_test = utils.get_dataset(task_name=task_name, with_splits=True)
    # Random samples
    random_samples_shape = (1000, x_train.shape[1])
    x_random = np.random.randint(low=0, high=2, size=random_samples_shape)

    # Get predictions from Fourier
    pred_fourier_test = fourier_transform[x_test]
    pred_fourier_train = fourier_transform[x_train]
    pred_fourier_random = fourier_transform[x_random]
    # Get predictions from neural net
    pred_nn_test = get_neural_net_predictions(task_name, x_test, save_to_cache=False)
    pred_nn_train = get_neural_net_predictions(task_name, x_train, save_to_cache=False)
    pred_nn_random = get_neural_net_predictions(task_name, x_random, save_to_cache=False)

    metric_dict = {
    "fourier_test_r2": r2_score(y_test, pred_fourier_test),
    "fourier_train_r2": r2_score(y_train, pred_fourier_train),
    "nn_test_r2": r2_score(y_test, pred_nn_test),
    "nn_train_r2": r2_score(y_train, pred_nn_train),
    "random_r2": r2_score(pred_nn_random, pred_fourier_random)
    }

    return metric_dict


def extract_fourier_and_evaluate(task_name):
    dataset_settings = utils.get_dataset_settings()
    k, degree = dataset_settings["k"][task_name], dataset_settings["degree"][task_name]
    time_samples_config = {"k": k, "degree": degree}
    time_samples = get_fourier_time_samples_with_dummy_function(task_name, k, degree)
    predictions = get_neural_net_predictions(task_name, time_samples, save_to_cache=True,
                                             time_samples_config=time_samples_config)
    fourier_transform = compute_fourier(task_name, predictions, save_to_cache=True,
                                        time_samples_config=time_samples_config)
    # Wrapper class for faster evaluations
    fourier_transform = Fourier(fourier_transform)
    metrics = test_fourier_quality(task_name, fourier_transform)
    return fourier_transform, metrics


if __name__ == "__main__":
    task_name = "sgemm"
    #task_name = "entacmaea"
    fourier_transform, metrics = extract_fourier_and_evaluate(task_name)
