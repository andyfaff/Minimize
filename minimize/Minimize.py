import numpy as np


class Minimize(object):
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        pass

    def solve(self):
        # repeatedly call __next__ until you are converged
        pass

    @property
    def x(self):
        "The Solution vector"
        pass

    def converged(self):
        # the truth of whether a solve has finished
        pass

    @property
    def result(self):
        # the OptimizeResult associated with the minimizer
        pass