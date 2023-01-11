import abc


# This is an abstract class for implementing your own compressed sensing over finite fields algorithm


class FiniteFieldCS(abc.ABC):

    @abc.abstractmethod
    def __init__(self, degree, n):
        pass

    @abc.abstractmethod
    def get_measurement_matrix(self):
        pass

    @abc.abstractmethod
    def recover_low_degree_vector(self, measurement_binary):
        pass
