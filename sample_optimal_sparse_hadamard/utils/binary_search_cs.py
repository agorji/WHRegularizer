import numpy as np
from math import floor, ceil, log


class BinarySearchCS:

    def __init__(self, n, **kwargs):
        # Code block length
        # The default values are for graph cut functions i.e degree = 2
        # no_bins is the number of bins the coordinates are hashed to in the first round
        # no of iterations is the number of times the recursive procedure is called
        # ratio is the bucket change ratio in each of the rounds of the recursive procedure
        self.n = n
        # No. of bins coordinates are hashd to in the first round
        try:
            self.no_bins = kwargs["no_bins"]
        except KeyError:
            self.no_bins = 3
        # No. of iterations
        try:
            self.iterations = kwargs["iterations"]
        except KeyError:
            self.iterations = 1
        # Halving ration of the buckets
        try:
            self.ratio = kwargs["ratio"]
        except KeyError:
            self.ratio = 2
        # List of all the measurements to be put into the measurement matrix
        self.measurements = []
        # A dictionary format of the measurements to be used in the recovery process
        self.recovery_dict = {}
        B = self.no_bins
        # Index keeps track of the number of the current measurement
        index = 0
        self.coord_map = {}
        for iter in range(self.iterations):
            self.recovery_dict[iter] = {}
            # hash coordinates to bins
            self.coord_map[iter] = {}
            for j in range(self.n):
                bin = np.random.randint(0, B)
                try:
                    self.coord_map[iter][bin].append(j)
                except KeyError:
                    self.coord_map[iter][bin] = [j]
                    self.recovery_dict[iter][bin] = []
            # print(self.coord_map[iter])
            for bin, coordinates in self.coord_map[iter].items():
                # Subsample Signal #
                # This coressponds to running a binary search on the coordinates
                # in the given bin
                bitRange = list(range((int(ceil(log(len(coordinates), 2))))))
                bitRange.append("all_ones")
                # print("bitrange=", bitRange)
                for bit in bitRange:
                    # bit refers to which bit of the location
                    # of the single 1 this binary search will specify
                    shift = self.__computeShift(coordinates, bit)
                    self.measurements.append(shift)
                    self.recovery_dict[iter][bin].append((index, bit))
                    index += 1
            B = int(floor(B / self.ratio))
        # Construct measurement matrix
        self.no_binary_measurements = len(self.measurements)
        self.measurement_matrix = np.zeros((self.no_binary_measurements, self.n), dtype=int)
        for row in range(self.no_binary_measurements):
            self.measurement_matrix[row, :] = self.measurements[row]

    def get_measurement_matrix(self):
        return self.measurement_matrix

    def recover_vector(self, measurement_binary, bucket=None):
        # print("Measurement_binary=", measurement_binary, "measurmenet_matrix", self.get_measurement_matrix())
        current_estimate = np.zeros(self.n, dtype=int)
        for iter in range(self.iterations):
            # print("iter=", iter, "current_estimate", current_estimate)
            residual_estimate = np.zeros(self.n, dtype=int)
            for bin in self.recovery_dict[iter]:
                coordinates = self.coord_map[iter][bin]
                # print("bin=", bin, "coordinates=", coordinates)
                recovered_bit_index = 0
                for index, bit in self.recovery_dict[iter][bin]:
                    residual_measurement = (measurement_binary[index] + np.dot(self.measurement_matrix[index],
                                                                               current_estimate)) % 2
                    # print("index=", index, "residual_measurement=", residual_measurement, "bit=", bit)
                    if bit == "all_ones":
                        all_ones_bit = residual_measurement
                        continue
                    recovered_bit_index += (2 ** bit) * residual_measurement
                cond_1 = (recovered_bit_index == 0)
                cond_2 = (all_ones_bit == 1)
                if cond_1 and cond_2:
                    residual_estimate[coordinates[0]] = 1
                    # print("Enterd if")
                elif not cond_1:
                    try:
                        residual_estimate[coordinates[recovered_bit_index]] = 1
                    except IndexError:
                        None
                # print("residual_estimate=", residual_estimate)
            current_estimate = (current_estimate + residual_estimate) % 2
        return tuple(current_estimate)

    def __computeShift(self, coordinates, bit):
        length = len(coordinates)
        if bit == "all_ones":
            a = np.ones(length, dtype=int)
        else:
            a = np.arange(length)
            a = np.floor(a / (2 ** bit))
            a = (1 - (-1) ** a) / 2
            a = a.astype(int)
        shift = np.zeros(self.n, dtype=int)
        shift[coordinates] = a
        return shift


if __name__ == "__main__":
    np.random.seed(9)
    binary_search_cs = BinarySearchCS(10)
    print(binary_search_cs.measurements, "\n", binary_search_cs.recovery_dict, "\n", )
    print(binary_search_cs.recover_vector(
        np.dot(binary_search_cs.get_measurement_matrix(), [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]) % 2))
