from utils.hashing import Hashing
import numpy as np
from functools import reduce
import operator as op
from math import ceil
from pyscipopt import Model, quicksum
import unittest


class RandomCS:
    def __init__(self, n, hash, degree, sampling_factor):
        assert (hash.n == n)
        self.n = n
        self.degree = degree
        self.sampling_factor = sampling_factor
        # Total number of measurements needed based on combinatorial bound
        self.no_binary_measurements = int(sampling_factor * RandomCS._get_number_measurements(n, degree))
        no_extra_measurments = self.no_binary_measurements - hash.b
        assert(no_extra_measurments > 0)
        extra_measurements = np.random.randint(low=0, high=2, size=(n, no_extra_measurments))
        self.measurement_matrix = np.hstack([hash.P, extra_measurements]).transpose()

    def get_measurement_matrix(self):
        return self.measurement_matrix

    def recover_vector(self, measurement_binary, bucket=None):
        if len(measurement_binary) != self.no_binary_measurements:
            raise ValueError("Bin or measurement does not have the correct dimension")

        # Create a model instance
        model = Model()
        model.hideOutput()
        # Fix infeasibility bug
        model.setParam("presolving/maxrounds", 0)
        vars = [0] * self.n
        objective = None
        # Vars and objective
        for j in range(self.n):
            vars[j] = model.addVar(f'x_{j}', vtype="B")
            if j == 0:
                objective = vars[0]
            else:
                objective = objective + vars[j]
        # add the constraints for measurments
        for i in range(self.no_binary_measurements):
            cons = [vars[j] for j in range(self.n) if self.measurement_matrix[i][j] == 1]
            model.addConsXor(cons, True if measurement_binary[i] == 1 else False)
        # add extra constraint for degree
        model.addCons(quicksum(vars[j] for j in range(self.n)) <= self.degree)
        # Find solution
        model.optimize()
        sol = model.getBestSol()
        sol = np.array([int(sol[vars[j]]) for j in range(self.n)], dtype=int)
        return sol

    @staticmethod
    def _get_number_measurements(n, d):
        return 2* ceil(np.log2(sum([RandomCS._nCr(n,i) for i in range(0,d+1)]))) +1

    @staticmethod
    def _nCr(n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom


class TestRandomCS(unittest.TestCase):
    def test(self):
        n, b, degree = 10, 2, 5
        hash = Hashing(n,b)
        random_cs = RandomCS(n, hash, degree)
        # Initialize some random frequency
        frequency = np.array([0] * n, dtype=int)
        #TODO:Frequency might not actually be degree exactly 4 since we are doing sampling with replacement
        support = np.array([np.random.randint(0, n) for _ in range(degree)], dtype=int)
        frequency[support] = 1
        y = np.dot(random_cs.measurement_matrix, frequency) % 2
        result = random_cs.recover_vector(y)
        self.assertTrue(np.array_equal(frequency, result))



if __name__ == "__main__":
    unittest.main()