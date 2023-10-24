"""
@author Antonio Bruce Webb(Organi)
"""

from FunctionParser import FunctionParser


class LeakyRELUActivationFunction:

    def __init__(self, alpha, maximum_activation):
        self.alpha = alpha
        self.maximum_activation = maximum_activation

    def calcualate(self, x):
        if x <= 0 and not x > self.maximum_activation:
            return ((x * self.alpha) / self.maximum_activation)

        elif x >= self.maximum_activation:
            return 1
        else:
            return (x / self.maximum_activation)

