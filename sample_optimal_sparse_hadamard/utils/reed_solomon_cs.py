from math import floor, ceil
import reedsolo as rs
import numpy as np


class ReedSolomonCS:

    def __init__(self, n, degree):
        # Code block length
        self.n = n
        self.degree = degree
        # Finite field of 2 ** (self.p)
        self.p = 0
        while 2 ** self.p <= n:
            self.p += 1
        if 2 * degree >= n:
            raise ValueError("2*d hast to be strictly less than n")
        # Want to locate/correct up to d errors
        self.no_check_symbols = 2 * degree
        self.no_message_symbols = n - 2 * degree
        prim = rs.find_prime_polys(c_exp=self.p, fast_primes=True, single=True)
        rs.init_tables(c_exp=self.p, prim=prim)
        self.generator_polynomial = rs.rs_generator_poly_all(n)[self.no_check_symbols]

        # Extract parity check matrix
        self.parity_check_matrix = np.zeros(shape=(self.no_check_symbols, n), dtype=np.uint16)

        for i in range(n):
            codeword = [0] * n
            codeword[i] = 1
            codeword = bytearray(codeword)
            parity_column = rs.rs_calc_syndromes(codeword, self.no_check_symbols)
            self.parity_check_matrix[:, i] = parity_column[1:]
        # print(self.parity_check_matrix)
        # Convert the parity check matrix into a binary version
        self.no_binary_measurements = self.p * self.no_check_symbols
        self.parity_check_matrix_binary = np.zeros(shape=(self.no_binary_measurements, n), dtype=np.uint16)
        for i in range(self.no_check_symbols):
            for j in range(n):
                self.parity_check_matrix_binary[i * self.p:(i + 1) * self.p, j] = self._to_binary(
                    self.parity_check_matrix[i, j])
        # print(self.parity_check_matrix_binary)

    def _to_binary(self, number):
        output_as_list = [int(x) for x in '{:0{size}b}'.format(number, size=self.p)]
        output_as_list.reverse()
        return np.array(output_as_list, dtype=np.uint16)

    def _to_finite_field(self, binary_vector):
        return np.sum(np.multiply(binary_vector, 2 ** np.arange(self.p, dtype=np.uint16)))

    def get_measurement_matrix(self):
        return self.parity_check_matrix_binary

    def recover_vector(self, measurement_binary, bucket=None):
        measurement_binary = np.array(measurement_binary)
        if measurement_binary.shape != (self.no_binary_measurements,):
            raise ValueError("The measurement vector does not have the correct dimension")
        # First we convert the measurement vector from binary to the finite field
        syndrome = [0] * self.no_check_symbols
        for i in range(self.no_check_symbols):
            syndrome[i] = self._to_finite_field(measurement_binary[i * self.p:(i + 1) * self.p])
        syndrome = [0] + syndrome
        # print(syndrome)
        # This part mimics the rs_correct_msg function
        if max(syndrome) == 0:
            return tuple([0] * self.n)
        # compute the Forney syndromes, which hide the erasures from the original syndrome (so that BM will just have to deal with errors, not erasures)
        fsynd = rs.rs_forney_syndromes(syndrome, [], self.n)
        # compute the error locator polynomial using Berlekamp-Massey
        err_loc = rs.rs_find_error_locator(fsynd, self.no_check_symbols)
        # locate the message errors using Chien search (or bruteforce search)
        err_pos = rs.rs_find_errors(err_loc[::-1], self.n)
        recovered_freq = np.zeros(self.n, dtype=int)
        recovered_freq[err_pos] = 1
        return tuple(recovered_freq)

if __name__ == "__main__":
    n = 20
    d = 4
    a = ReedSolomonCS(n, d)
    low_degree_vector = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    measurement_bin = np.dot(a.get_measurement_matrix(), low_degree_vector) % 2
    print(measurement_bin)
    print(a.recover_vector(measurement_bin))
    print("haha")
