"""
@author Antonio Bruce Webb(Organi)
"""
from math import e
from math import pow

class HyperbolicTangentActivationFunction:

    def __init__(self):
        self.function = "(e^x - e^-x) / (e^x + e^-x)"

    @staticmethod
    def calculate(x):
        return (pow(e, x) - (pow(e, -x))) / (pow(e, x) + (pow(e, -x)))

    def comp_derivative(self, weighted_sum, del_cost):
        return 1 - ((self.calculate(weighted_sum))**2)
