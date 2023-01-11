import numpy as np
from math import ceil, log, isclose, floor
from utils import hashing
from utils import naive_cs, binary_search_cs, reed_solomon_cs, efficient_hashing_based_cs
from utils.WHT import WHT
from utils.random_function import RandomFunction
from utils.cosamp import WHTAlgorithm

class SWHT(object):
    def __init__(self, n, K, C=4, ratio=1.4, finite_field_class="naive_cs",  **kwargs):
        # C bucket constant
        self.C = C
        # Size of Ground set
        self.n = n
        # Sparsity
        self.K = K
        # Bucket halving ratio
        self.ratio = ratio
        # What to use for the finite field decoding
        self.finite_field_class = finite_field_class
        # Settings for the finite field decoding
        self.settings_finite_field = kwargs

    def run(self, x):
        # B = no of bins we are hashing to
        B = int(self.K * self.C)
        b = int(ceil(log(B, 2)))
        # Number of iterations for peeling
        T = int(floor(log(B, self.ratio))) - 1
        # current_estimate will hold as key frequencies and as value amplitudes
        current_estimate = {}
        for i in range(T):
            # Define a new hashing matrix A
            if self.finite_field_class in ["efficient_hashing_based_cs", "random]"]:
                hash = hashing.InvertibleHashing(self.n, b)
            else:
                hash = hashing.Hashing(self.n, b)
            # hashed_estimate will hold as keys bin frequencies and as values
            # tuples where the first element is the freq hashed to that bin
            # and second element is the amplitude
            hashedEst = self.hash_frequencies(hash, current_estimate)
            residual_estimate = self.detect_frequency(x, hash, hashedEst, b)

            ##########################
            # Run iterative updates
            for freq in residual_estimate:
                if freq in current_estimate:
                    current_estimate[freq] = current_estimate[freq] + residual_estimate[freq]
                    if isclose(current_estimate[freq], 0.0, abs_tol=0.0001):
                        del current_estimate[freq]

                else:
                    current_estimate[freq] = residual_estimate[freq]

            # Buckets sizes for hashing reduces by half for next iteration
            B = int(ceil(B / self.ratio))
            b = int(ceil(log(B, 2)))
        return current_estimate

    def hash_frequencies(self, hash, est):
        # This function hashes the current estimated frequencies
        # of the signal to the buckets
        hashed_estimate = {}
        for key in est:
            hashed_key = hash.do_FreqHash(key)
            if hashed_key not in hashed_estimate:
                #  Initialize empty list
                hashed_estimate[hashed_key] = []
            hashed_estimate[hashed_key].append((key, est[key]))
        return hashed_estimate

    def __get_finite_field_recovery(self, hash):
        if self.finite_field_class == "naive_cs":
            return naive_cs.NaiveCS(self.n)
        elif self.finite_field_class == "efficient_hashing_based_cs":
            return efficient_hashing_based_cs.EfficientHashingBasedCS(self.n, hash)
        elif self.finite_field_class == "reed_solomon_cs":
            try:
                return reed_solomon_cs.ReedSolomonCS(self.n, self.settings_finite_field["degree"])
            except KeyError:
                print("For using reed_solomon decoding you need to specify the degree")
        elif self.finite_field_class == "binary_search_cs":
            print("################")
            return binary_search_cs.BinarySearchCS(self.n, **self.settings_finite_field)
        else:
            raise ValueError("The finite_field_class \"", self.finite_field_class, "\"does not exist")
    def get_WHT(self, x, b):
        if self.WHT_algorithm == "normal":
            return WHT(x)
        elif self.WHT_algorithm == "cosamp":
            WHT_algorithm = WHTAlgorithm(b, k = (2**b)/self.C ,C=1, algorithm="cosamp")
            out = WHT_algorithm.run(x)
            return out

    def detect_frequency(self, x, hash, hashedEst, b):

        # We need the WHT with shift 0 for reference
        a = np.zeros((self.n), dtype=np.int64)
        hashedSignal = hash.do_TimeHash(x, a)
        ref_signal = self.get_WHT(hashedSignal,b)
        # print("hash=", hash.P.transpose())
        # print("ref_signal", ref_signal)
        # This dictionary will hold the WHTs of the subsampled signals with the measurements
        hashedWHT = {}
        finite_field_cs = self.__get_finite_field_recovery(hash)
        # Subsample Signal
        no_measurements = finite_field_cs.no_binary_measurements
        for j in range(no_measurements):
            a = finite_field_cs.get_measurement_matrix()[j, :]
            # print("Measurement=", a)
            hashedSignal = hash.do_TimeHash(x, a)
            hashedWHT[j] = self.get_WHT(hashedSignal,b)
            # print("WHT of measurement=", hashedWHT[j])

        # Detect Frequencies
        # Dictionary of detected frequencies
        detectedFreq = {}
        # i is the number of the bucket we are checking in the iterations below
        # print("hashedEst=", hashedEst)
        for i in range(hash.B):
            bucket = self.toBinaryTuple(i, hash.b)
            # print("bucket", bucket)
            # Compute the values of the current estimation of signal hashed to this bucket and subtract it off
            if bucket in hashedEst:
                for X in hashedEst[bucket]:
                    ref_signal[bucket] = ref_signal[bucket] - X[1]
            # print("In bucket ref_signal=", ref_signal)
            # Only continue if a frequency with non-zero amplitude is hashed to bucket j
            # print("checking ref_signal", ref_signal[bucket])
            if isclose(ref_signal[bucket], 0.0, abs_tol=0.001):
                # print("Entered if statement for ref_signal[bucket]=0")
                continue
            # Subtract the current hashed estimates signal from each
            # of the buckets
            for j in range(no_measurements):
                if bucket in hashedEst:
                    for X in hashedEst[bucket]:
                        if self.__inp(X[0], finite_field_cs.get_measurement_matrix()[j, :]) == 0:
                            hashedWHT[j][bucket] = hashedWHT[j][bucket] - X[1]
                        else:
                            hashedWHT[j][bucket] = hashedWHT[j][bucket] + X[1]
                # freq is the frequecy preset in this bin
                # print("In bucket measurement signal =", hashedWHT[j])
            measurement = [0] * no_measurements
            for j in range(no_measurements):
                if np.sign(hashedWHT[j][bucket]) != np.sign(ref_signal[bucket]):
                    measurement[j] = 1
                else:
                    measurement[j] = 0
            # print("Measurement vector is", measurement)
            try:
                recovered_freq = finite_field_cs.recover_vector(measurement, bucket)
            except:
                continue
            detectedFreq[recovered_freq] = ref_signal[bucket]
        # print (ref_signal)
        return detectedFreq

    # This function computes the inner product of two 0-1 n-tuples
    @staticmethod
    def _inp(a, b):
        # print("inp", size(a), size(b))
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) % 2

    def toBinaryTuple(self, i, b):
        # Converts integer i into an b-tuple of 0,1s
        a = list(bin(i)[2:].zfill(b))
        a = tuple([int(x) for x in a])
        return a


if __name__ == "__main__":
    n = 10
    k = 1
    swht = SWHT(n, k)
    # swht = SWHT(n, k, 8, 2, finite_field_class="reed_solomon_cs", degree=5)
    # swht = SWHT(n, k, 1.4, 1.4, finite_field_class="binary_search_cs", no_bins=10, iterations=2)
    # swht = SWHT(n, k, 1.4, 1.4, finite_field_class="hashing_based_cs")
    f = RandomFunction(n, k, 5)
    print("f is :", f, flush=True)
    out = swht.run(f)
    print("out is", out)
    fprime = RandomFunction.create_from_FT(n, out)  #  print("fprime is ", fprime)
    if (f == fprime):
        print("Success")
    print(f.get_sampling_complexity())
