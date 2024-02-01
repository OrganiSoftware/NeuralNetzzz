"""
@author Antonio Bruce Webb(Organi)
"""

from FunctionParser import FunctionParser
from DualNumber import DualNumber

class LeakyRELUActivationFunction:

    def __init__(self, alpha, maximum_activation):
        self.alpha = alpha
        self.alpha_dual = DualNumber(alpha, 0)
        self.maximum_activation = maximum_activation
        self.max_activation_dual = DualNumber(maximum_activation, 0)


    def calculate(self, x):
        if x <= 0:
            if x * self.alpha <= -self.maximum_activation:
                resultant = -1
            else:
                resultant = (x * self.alpha)/self.maximum_activation
        else:
            if x >= self.maximum_activation:
                resultant = 1
            else:
                resultant = x / self.maximum_activation
        return resultant

    def comp_partial(self, x, dual_num):
        dual_partial = None
        if x <= 0:
            if x * self.alpha <= -self.maximum_activation:
                dual_partial = dual_num - (dual_num + self.max_activation_dual)
            else:
                dual_partial = (self.alpha_dual * dual_num)/self.max_activation_dual
        else:
            if x >= self.maximum_activation:
                dual_partial = dual_num - (dual_num - self.max_activation_dual)
            else:
                dual_partial = dual_num /self.max_activation_dual
        return dual_partial
