'''
SPRIGHT sampling derived from EpistaticNet Github repo:
https://github.com/amirmohan/epistatic-net/blob/main/wht-sampling.ipynb
'''

from sre_constants import RANGE
from matplotlib.widgets import RangeSlider
import numpy as np
import pickle

import os
import glob

def random_binary_matrix(m, n, random_generator, p=0.5):
    A = random_generator.binomial(1,p,size=(m,n))
    return A

def dec_to_bin(x, num_bits):
    assert x < 2**num_bits, "number of bits are not enough"
    u = bin(x)[2:].zfill(num_bits)
    u = list(u)
    u = [int(i) for i in u]
    return np.array(u)

def get_sampling_index(x, A, p=0):
    """
    x: sampling index
    A: subsampling matrix
    p: delay
    """
    num_bits = A.shape[0]
    x = dec_to_bin(x, num_bits)
    r = x.dot(A) + p
    return r % 2

def get_random_binary_string(num_bits, random_generator, p=0.5):
    a = random_generator.binomial(1,p,size=num_bits)
    return a

def random_delay_pair(num_bits, target_bit, random_generator):
    """
    num_bits: number of bits
    location_target: the targeted location (q in equation 26 in https://arxiv.org/pdf/1508.06336.pdf)
    """
    e_q = 2**target_bit
    e_q = dec_to_bin(e_q, num_bits)
    
    random_seed = get_random_binary_string(num_bits, random_generator)
    
    return random_seed, (random_seed+e_q)%2

def make_delay_pairs(num_pairs, num_bits, random_generator):
    z = []
    # this is the all zeros for finding the sign
    # actually we do not need this here because we solve
    # a linear system to find the value of the coefficient
    # after the location is found -- however, i am going to
    # keep this here not to have to change the rest of the code
    # that takes delays of this form
    z.append(dec_to_bin(0,num_bits))
    # go over recovering each bit, we need to recover bits 0 to num_bits-1
    for bit_index in range(0, num_bits):
        # we have num_pairs many pairs to do majority decoding
        for pair_idx in range(num_pairs):
            a,b = random_delay_pair(num_bits, bit_index, random_generator)
            z.append(a)
            z.append(b)
    return z


class SPRIGHTSample:
    def __init__(self, n, m, d, random_seed=0, repetition=3, use_cache=True) -> None:
        '''
            Args:
                - m: the sparsity we target is around K = 2**m
                - n: this is the signal length N = 2**n
                - d: num delays per single bit of the location index
                    (the larger this number the more tolerant to noise we are)
                    so one needs to play around with this a bit

        '''
        self.n = n
        self.m = m
        self.d = d

        self.sampling_matrices = []
        self.delay_matrices = []
        self.sampling_locations = []

        sample_name = f'N{n}-m{m}-d{d}-seed{random_seed}'
        sample_dir = f'{os.environ.get("SPRIGHT_CACHE") if "SPRIGHT_CACHE" in os.environ else "cache"}/{sample_name}'

        # Load from cache if possible
        if use_cache:
            if len(list(glob.glob(sample_dir))) > 0:
                print("Loading SPRIGHT samples from cache ...")
                for file in list(glob.glob(f'{sample_dir}/sampling-matrix-*')):
                    self.sampling_matrices.append(pickle.load(open(file,'rb')))
                for file in list(glob.glob(f'{sample_dir}/delays-*')):
                    self.delay_matrices.append(pickle.load(open(file,'rb')))
                for file in list(glob.glob(f'{sample_dir}/sampling-locations-*')):
                    self.sampling_locations.append(pickle.load(open(file,'rb')))
                
                return

        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)

        # Create a numpy random generator for reproducibility
        rng = np.random.default_rng(random_seed)

        # total number of samples from the signal
        # from this you can calculate the time necessary
        # you can adjust d accordingly to tune the time necessary
        # the larger d is better, but then it takes more time too
        total_samples = (2**m)*n*(d*2+1)

        # We need to run the code below 3 times and save as separate matrices
        for run_number in range(repetition):
            A = random_binary_matrix(m, n, rng)

            sampling_locations_base = []

            for i in range(2**A.shape[0]):
                sampling_locations_base.append(get_sampling_index(i,A))
            sampling_locations_base = np.array(sampling_locations_base)

            delays = make_delay_pairs(d, A.shape[1], rng)

            # This is a list of all matrices of all sampling locations necessary
            all_sampling_locations = []

            for current_delay in delays:
                new_sampling_locations = (sampling_locations_base + current_delay) % 2
                all_sampling_locations.append(new_sampling_locations)
            
            self.sampling_matrices.append(A)
            self.delay_matrices.append(delays)
            self.sampling_locations.append(all_sampling_locations)

            # Cache calculated matrices
            pickle.dump(A, open(f"{sample_dir}/sampling-matrix-{run_number}.p", "wb" ) )
            pickle.dump(delays, open(f"{sample_dir}/delays-{run_number}.p", "wb" ) )
            pickle.dump(all_sampling_locations, open(f"{sample_dir}/sampling-locations-{run_number}.p", "wb"))

if __name__ == "__main__":
    sample = SPRIGHTSample(13, 4, 3)
    print(len(sample.sampling_locations))
    print(sample.sampling_matrices[0].shape)
    print(len(sample.delay_matrices[0]))
    print(len(sample.sampling_locations[0]))



        