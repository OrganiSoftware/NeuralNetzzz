"""
@author Antonio Bruce Webb(Organi)
"""
from math import e
from math import pow


class SigmoidalActivationFuction:

    def __init__(self):
        self.function = "(1/(1+e^-x))"

    @staticmethod
    def calculate(x):
        return 1/(1+pow(e, -x))

    def function_string(self):
        return self.function

