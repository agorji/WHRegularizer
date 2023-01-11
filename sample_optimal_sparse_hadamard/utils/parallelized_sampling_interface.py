import numpy as np
import pickle as pkl
import os

class DummyFunction:
    """ A dummy function that always returns zero and caches all the time queries that was made to it
    Attributes
    ----------
    samples : list
        list containing samples that were queried from the function

    """

    def __init__(self):
        self.samples = []

    def __getitem__(self, t_list):
        self.samples.append(t_list)
        return np.zeros(len(t_list))

    def get_time_samples(self):
        """returns list containing samples that were queried from the function
        """
        return self.samples

    def export_times_samples_to_file(self, dir, filename):
        """ dumps the list of time samples to a pkl file
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = f"{dir}/{filename}"
        with open(path, "wb") as f:
            pkl.dump(self.samples, f)


class FastFunction:
    """ A cached function that is fast
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        path or value_list:
            Either a path containing a pkl files where list of values are located or a list of values
        """
        if "path" in kwargs:
            with open(kwargs["path"], "rb") as f:
                self.values = pkl.load(f)
        elif "pred_array" in kwargs:
            self.values = kwargs["pred_array"]
        else:
            raise ValueError("You must pass either a path or value_array argument to the initializer")
        self.no_samples = len(self.values)
        self.index = 0

    def reset(self):
        self.index = 0

    def __getitem__(self, item):
        try:
            ret_value = np.array(self.values[self.index: self.index + item.shape[0]])
            self.index += item.shape[0]
            return ret_value
        except IndexError:
            raise IndexError("Index ran out of bounds. More values queried from the function "
                             "than was available in the cache")
