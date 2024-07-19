"""
@author Antonio Bruce Webb(Organi)
"""
from math import e
from math import pow
from DualNumber import  DualNumber


class SigmoidalActivationFuction:

    def __init__(self):
        self.dual_e = DualNumber(e, 0)
        self.function = "(1/(1+e^-x))"

    @staticmethod
    def calculate(x):
        return 1/(1+pow(e, -x))

    def comp_partial(self, x, dual_num):
        one = DualNumber(1, 0)
        neg_one = DualNumber(-1, 0)
        negative_dual_num = neg_one * dual_num
        new_dual_num = one / (one + self.dual_e.dual_pow_dual(negative_dual_num))
        return new_dual_num

    def comp_derivative(self, weighted_sum):
        return (e**(weighted_sum * -1))/((1 + e**(weighted_sum * -1))**2)