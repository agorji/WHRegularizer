import numpy as np


class Fourier:
    """A class that maintains the Fourier transform and allows us to efficiently evaluate it
    """

    def __init__(self, tuple_fourier_transform):
        if len(tuple_fourier_transform) == 0:
            tuple_fourier_transform = {(0,) : 0}
        self.n_vars = len(next(iter(tuple_fourier_transform.keys())))
        self.k = len(tuple_fourier_transform)
        self.freq_matrix = np.zeros((self.k, self.n_vars), dtype=int)
        self.amp_matrix = np.zeros(self.k)
        # frozen set representation fo the Fourier transform
        self.set_fourier_transform = {}
        for i, (freq, amplitude) in enumerate(tuple_fourier_transform.items()):
            self.freq_matrix[i] = freq
            self.amp_matrix[i] = amplitude
            self.set_fourier_transform[self.__get_set(freq)] = amplitude

    @staticmethod
    def __get_set(freq):
        freq = list(freq)
        # print("freq", freq)
        index = 0
        set = frozenset()
        for i in freq:
            if i >= 1:
                set = set.union([index])
            index += 1
        # print("set", set)
        return set


    def __getitem__(self, x):
        if len(x.shape) ==1:
            x = np.reshape(x, newshape=(1, x.shape[0]))

        ret_value = (-1) ** (((x @ self.freq_matrix.T) % 2).squeeze())
        ret_value = (ret_value  @ self.amp_matrix).squeeze()
        return ret_value


