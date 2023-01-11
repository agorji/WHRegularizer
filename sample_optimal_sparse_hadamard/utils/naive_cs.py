import numpy as np

class NaiveCS:

    def __init__(self, n):
        # Code block length
        self.n = n
        # Finite field of 2 ** (self.p)
        self.no_binary_measurements = n
        self.measurement_matrix = np.identity(n, dtype=int)

    def get_measurement_matrix(self):
        return self.measurement_matrix

    def recover_vector(self, measurement_binary, bucket=None):
        return tuple(measurement_binary)

if __name__ == "__main__":
    naive_cs = NaiveCS(3)
    print(naive_cs.get_measurement_matrix())
    measurement = (0, 0, 1)
    print(naive_cs.recover_vector(measurement))
