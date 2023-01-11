import numpy as np
from math import ceil, log, isclose, floor
from sample_optimal_sparse_hadamard.utils import hashing
from sample_optimal_sparse_hadamard.utils.WHT import WHT
from sample_optimal_sparse_hadamard.utils.random_function import RandomFunction


class SWHTRobust(object):
    def __init__(self, n, K, C=4 , ratio=1.3):
        # C bucket constant
        self.C = C
        # Size of Ground set
        self.n = n
        # Sparsity
        self.K = K
        # Bucket halving ratio
        self.ratio = ratio

    def run(self, x):
        # B = no of bins we are hashing to
        # B = 48 * self.K
        B = int(self.K * self.C)
        # print("B=", B)
        b = int(ceil(log(B, 2)))
        # T = no. of iterations
        # T = int(min(floor(log(B, 2)) - 1, ceil(log(self.K, 2)) + 1))
        # T = ceil(log(self.K,4))
        # This should be minus 1
        T = int(floor(log(B, self.ratio))) - 1
        print(T)
        # T = int(min(floor(log(B, 1.6)) - 1, 10*ceil(log(self.K, 2)) + 1))
        # print(T)
        # current_estimate will hold as key frequencies and as value amplitudes
        current_estimate = {}
        for i in range(T):
            print("Iteration ", i)
            # print("B=", B, "b=", b)
            # Define a new hashing matrix A
            hash = hashing.Hashing(self.n, b)
            # hashedEstimate will hold as keys bin frequencies and as values
            # tuples where the first element is the freq hashed to that bin
            # and second element is the amplitude
            hashed_current_estimate = self.hashFrequencies(hash, current_estimate)
            new_estimate = self.detectFrequency(x, hash, hashed_current_estimate)

            #########################
            # x.statistic(detectedFreq)
            # bucketCollision = {}
            # for edge in x.graph:
            #     freq = np.zeros((self.n))
            #     freq[edge[0]] = 1
            #     freq[edge[1]] = 1
            #     freq = tuple(freq)
            #     print(edge, "hashed to ", hash.do_FreqHash(freq))
            #     try:
            #         bucketCollision[hash.do_FreqHash(freq)].append(edge)
            #     except KeyError:
            #         bucketCollision[hash.do_FreqHash(freq)] = []
            #         bucketCollision[hash.do_FreqHash(freq)].append(edge)
            # collisions = 0
            # for bucket in bucketCollision:
            #     if len(bucketCollision[bucket]) > 1:
            #         collisions += len(bucketCollision[bucket])
            #         print(bucketCollision[bucket])
            # print("collisions=", collisions)
            ##########################
            # Run iterative updates
            for freq in new_estimate:
                if freq in current_estimate:
                    current_estimate[freq] = current_estimate[freq] + new_estimate[freq]
                    if isclose(current_estimate[freq], 0.0, abs_tol=0.001):
                        # print("deleting", freq)
                        del current_estimate[freq]

                else:
                    current_estimate[freq] = new_estimate[freq]

            # Buckets sizes for hashing reduces by half for next iteration
            B = int(ceil(B / self.ratio))
            b = int(ceil(log(B, 2)))
        return current_estimate

    def hashFrequencies(self, hash, est):
        # This function hashes the current estimated frequencies
        # of the signal to the buckets
        hashedEstimate = {}
        for key in est:
            hashed_key = hash.do_FreqHash(key)
            if hashed_key not in hashedEstimate:
                #  Initialize empty list
                hashedEstimate[hashed_key] = []
            hashedEstimate[hashed_key].append((key, est[key]))
        return hashedEstimate

    def detectFrequency(self, x, hash, hashed_current_estimate):

        no_trials = 20
        # print("hashing matrix is:", hash.P)
        # print("hashed_current_estimate is:", hashed_current_estimate)
        freq_dict = {}
        ampl_dict = {}
        successful_tries = {}
        successful_try_random_shift = {}
        for i in range(hash.B):
            bucket = self.toBinaryTuple(i, hash.b)
            # the frequency discovered in this bin
            freq_dict[bucket] = [0] * self.n
            # list of amplitudes in this bin each corresponding to a different random shift
            ampl_dict[bucket] = []
            # The number of times the sum of frequencies mapped to the bin (with the random shifts) exceeded the threshold
            successful_tries[bucket] = 0
            # the corresponding shift of the successful try
            successful_try_random_shift[bucket] = []

        random_shift_list = [np.random.randint(low=0, high=2, size=(self.n,)) for _ in range(no_trials)]
        for random_shift in random_shift_list:
            # print("randomshift", random_shift)
            hashed_signal = hash.do_TimeHash(x, random_shift)
            # print("After Zero shift ", str(x.sampCplx))
            # print("hashed_signal=", hashedSignal)
            ref_signal = WHT(hashed_signal)
            # print("reference signal")
            # print(ref_signal)
            # This dictionary will hold the WHTs of the subsampled signals
            hashedWHT = {}
            # Subsample Signal
            for j in range(self.n):
                # set a = e_j
                # print("e=", j)
                e_j = np.zeros((self.n), dtype=np.int64)
                e_j[j] = 1
                # print(a)
                hashed_signal = hash.do_TimeHash(x, (e_j + random_shift) % 2)
                hashedWHT[j] = WHT(hashed_signal)
                # print("e_", j, hashedWHT[j])

            # i is the number of the bucket we are checking in the iterations below
            for i in range(hash.B):
                bucket = self.toBinaryTuple(i, hash.b)
                # print("Bucket", bucket)
                # Compute the values of the current estimation of signal hashed to this bucket and subtract it off the
                # reference signal
                if bucket in hashed_current_estimate:
                    for X in hashed_current_estimate[bucket]:
                        if self._inp(X[0], random_shift) == 0:
                            ref_signal[bucket] = ref_signal[bucket] - X[1]
                        else:
                            ref_signal[bucket] = ref_signal[bucket] + X[1]

                # Only continue if a frequency with non-zero amplitude is hashed to bucket j
                # print("cheching ref_signal", ref_signal[bucket])
                # print("ref_signal after subtraction", ref_signal)
                if isclose(ref_signal[bucket], 0.0, abs_tol=0.001):
                    # print("Entered if statement for ref_signal[bucket]=0")
                    continue
                else:
                    successful_tries[bucket] += 1
                    successful_try_random_shift[bucket].append(random_shift)
                # Subtract the values of the current estimation of signal hashed to this bucket and subtract it off the
                # signal with measurements
                for j in range(self.n):
                    e_j = np.zeros((self.n), dtype=np.int64)
                    e_j[j] = 1
                    if bucket in hashed_current_estimate:
                        for X in hashed_current_estimate[bucket]:
                            if self._inp(X[0], random_shift + e_j) == 0:
                                hashedWHT[j][bucket] = hashedWHT[j][bucket] - X[1]
                            else:
                                hashedWHT[j][bucket] = hashedWHT[j][bucket] + X[1]
                    # print("hashedWHT e_", j, "after subtraction", hashedWHT[j])
                # freq is the frequecy preset in this bin
                for j in range(self.n):
                    if np.sign(hashedWHT[j][bucket]) != np.sign(ref_signal[bucket]):
                        try:
                            freq_dict[bucket][j] += 1
                        except KeyError:
                            freq_dict[bucket][j] =  0
                ampl_dict[bucket].append(ref_signal[bucket])
                # print("detected_frequencies", detected_frequencies, "detected_amplitudes", detected_amplitudes)
            # print (ref_signal)


        new_signal_estimate = {}
        # Take majority vote for frequency and median for amplitudes
        # print(detected_frequencies, "\n\n", detected_amplitudes, "\n\n")
        for bucket in freq_dict:
            if successful_tries[bucket] == 0:
                continue
            # print(successful_tries[bucket])
            recovered_freq = [0] * self.n
            for j in range(self.n):
                if freq_dict[bucket][j] > successful_tries[bucket] / 2:
                    recovered_freq[j] = 1
                else:
                    # recovered_freq[j] = 0
                    pass
            if hash.do_FreqHash(recovered_freq)!=bucket:
                continue
            index = 0
            for random_shift in successful_try_random_shift[bucket]:
                if self._inp(recovered_freq, random_shift) == 1:
                    ampl_dict[bucket][index] = -ampl_dict[bucket][index]
                index += 1
            recovered_ampl = np.median(ampl_dict[bucket])
            new_signal_estimate[tuple(recovered_freq)] = recovered_ampl
        return new_signal_estimate

    # This function computes the inner product of two 0-1 n-tuples
    @staticmethod
    def _inp(a, b):
        # print("_inp", size(a), size(b))
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) % 2

    @staticmethod
    def toBinaryTuple(i, b):
        # Converts integer i into an b-tuple of 0,1s
        a = list(bin(i)[2:].zfill(b))
        a = tuple([int(x) for x in a])
        return a


if __name__ == "__main__":
    # np.random.seed(4)
    n = 8
    k = 4
    swht = SWHTRobust(n, k, 10, 1.2)

    f = RandomFunction(n, k)
    print("f is :", f)
    out = swht.run(f)
    print("out is", out)
    fprime = RandomFunction.create_from_FT(n, out)  #  print("fprime is ", fprime)
    if (f == fprime):
        print("Success")

    # g = Graph(swht.n, swht.K)
    # print(g)
    # p = 0
    # for i in range(1):
    #     # print(i)
    #     y = swht.run(g)
    #     try:
    #         g2 = Graph.create_from_FT(swht.n, y)
    #     except AssertionError:
    #         continue
    #     if g == g2:
    #         p = p+1
    #     g.cache = {}
    # print(p)

    # y = swht.run(g)
    # print(y, "\n", g)
    # y.pop(tuple([0]*swht.n))
    # g2 = Graph.create_from_FT(swht.n, y)
    # print(g == g2)
    # print("SamplingComplexity =", g.sampCplx)
    # bit = 2
    # j = np.arange(20)
    # j = np.floor(j/(2 ** bit))
    # a = (1 - (-1) ** j)/2
    # a = a.astype(int)
    # print(a)
    # print(a.shape)
    # mask = np.zeros(20, dtype=int)
    # mask[4:6] = 1
    # mask[0:2] = 1
    # mask[14:16] = 1
    # print(mask.shape)
    # print(mask)
    # r = np.multiply(a, mask).reshape(20, 1)
    # print(r, r.shape)
    # bitIndexRange = list(range((int(ceil(log(100, 2))))))
    # bitIndexRange.append("ref")
    # print(bitIndexRange)
    # for j in bitIndexRange:
    #     print(j)
