"""
@author Antonio Bruce Webb(Organi)
"""
from math import e
from math import pow
from DualNumber import DualNumber

class HyperbolicTangentActivationFunction:

    def __init__(self):
        self.dual_e = DualNumber(e, 0)
        self.function = "(e^x - e^-x) / (e^x + e^-x)"

    def calculate(self, x):
        return (pow(e, x) - (pow(e, -x))) / (pow(e, x) + (pow(e, -x)))

    def comp_partial(self, x, dual_num):
        neg_one = DualNumber(-1, 0)
        negative_dual_num = neg_one * dual_num
        partial = ((self.dual_e.dual_pow_dual(dual_num) - self.dual_e.dual_pow_dual(negative_dual_num)) /
                   (self.dual_e.dual_pow_dual(dual_num) + self.dual_e.dual_pow_dual(negative_dual_num)))

        return partial

    def comp_derivative(self, weighted_sum):
        return 1 - ((self.calculate(weighted_sum))**2)
