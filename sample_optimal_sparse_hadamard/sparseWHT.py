import numpy as np
from math import ceil, log, isclose, floor
import hashing
from WHT import WHT
from random_function import RandomFunction


class SWHT(object):
    def __init__(self, n, K, C=1.3, ratio=1.4):
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
        T = int(floor(log(B, self.ratio))) - 1
        # print(T)
        # T = int(min(floor(log(B, 1.6)) - 1, 10*ceil(log(self.K, 2)) + 1))
        # print(T)
        # current_estimate will hold as key frequencies and as value amplitudes
        current_estimate = {}
        for i in range(T):
            # print("Iteration ", i)
            # print("B=", B, "b=", b)
            # Define a new hashing matrix A
            hash = hashing.Hashing(self.n, b)
            # hashedEstimate will hold as keys bin frequencies and as values
            # tuples where the first element is the freq hashed to that bin
            # and second element is the amplitude
            hashedEst = self.hashFrequencies(hash, current_estimate)
            new_estimate = self.detectFrequency(x, hash, hashedEst)

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

    def detectFrequency(self, x, hash, hashedEst):
        # We need the WHT with shift 0 for reference
        a = np.zeros((self.n), dtype=np.int64)
        # print("Before Zero shift ", str(x.sampCplx))
        hashedSignal = hash.do_TimeHash(x, a)
        # print("After Zero shift ", str(x.sampCplx))
        # print("hashed_signal=", hashedSignal)
        ref_signal = WHT(hashedSignal)
        # print("reference signal")
        # print(hashedWHT["ref"])
        # This dictionary will hold the WHTs of the subsampled signals
        hashedWHT = {}
        # Subsample Signal
        for j in range(self.n):
            # set a = e_j
            # print("e=", j)
            a = np.zeros((self.n), dtype=np.int64)
            a[j] = 1
            # print(a)
            hashedSignal = hash.do_TimeHash(x, a)
            hashedWHT[j] = WHT(hashedSignal)
            # print(hashedWHT[j].shape)

        # Detect Frequencies
        # Dictionary of detected frequencies
        detectedFreq = {}
        # i is the number of the bucket we are checking in the iterations below
        for i in range(hash.B):
            bucket = self.toBinaryTuple(i, hash.b)
            # Compute the values of the current estimation of signal hashed to this bucket and subtract it off
            if bucket in hashedEst:
                for X in hashedEst[bucket]:
                    ref_signal[bucket] = ref_signal[bucket] - X[1]

            # Only continue if a frequency with non-zero amplitude is hashed to bucket j
            # print("cheching ref_signal", ref_signal[bucket])
            if isclose(ref_signal[bucket], 0.0, abs_tol=0.0001):
                # print("Entered if statement for ref_signal[bucket]=0")
                continue
            # Subtract the current hashed estimates signal from each
            # of the buckets
            for j in range(self.n):
                if bucket in hashedEst:
                    for X in hashedEst[bucket]:
                        if (X[0][j] == 0):
                            hashedWHT[j][bucket] = hashedWHT[j][bucket] - X[1]
                        else:
                            hashedWHT[j][bucket] = hashedWHT[j][bucket] + X[1]
            # freq is the frequecy preset in this bin
            freq = [0] * self.n
            for j in range(self.n):
                if (np.sign(hashedWHT[j][bucket]) == np.sign(ref_signal[bucket])):
                    freq[j] = 0
                else:
                    freq[j] = 1
            detectedFreq[tuple(freq)] = ref_signal[bucket]
        # print (ref_signal)
        return detectedFreq

    # This function computes the inner product of two 0-1 n-tuples
    def inp(self, a, b):
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
    np.random.seed(4)
    n = 12
    k = 15
    swht = SWHT(n, k, 4, 1.4)

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
