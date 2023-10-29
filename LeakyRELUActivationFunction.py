"""
@author Antonio Bruce Webb(Organi)
"""

from FunctionParser import FunctionParser


class LeakyRELUActivationFunction:

    def __init__(self, alpha, maximum_activation):
        self.alpha = alpha
        self.maximum_activation = maximum_activation

    def calculate(self, x):
        if x <= 0 and not -self.maximum_activation > x:
            return (x * self.alpha)/self.maximum_activation
        elif -self.maximum_activation > x:
            return -1

        elif x >= self.maximum_activation:
            return 1
        else:
            return x / self.maximum_activation

