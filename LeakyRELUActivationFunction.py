"""
@author Antonio Bruce Webb(Organi)
"""

from FunctionParser import FunctionParser


class LeakyRELUActivationFunction:

    def __init__(self, alpha):
        self.alpha = alpha

    def calcualate(self, x):
        if x <= 0:
            return x * self.alpha
        else:
            return x


