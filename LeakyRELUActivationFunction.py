"""
@author Antonio Bruce Webb(Organi)
"""

from FunctionParser import FunctionParser


class LeakyRELUActivationFunction:

    def __init__(self, alpha, maximum_activation):
        self.alpha = alpha
        self.maximum_activation = maximum_activation

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

    def function_string(self, x):
        if x <= 0:
            if x * self.alpha <= -self.maximum_activation:
                function_string = "-1"
            else:
                function_string = "(" + str(x) + str(self.alpha) + ")/(" + str(self.maximum_activation) + ")"
        else:
            if x >= self.maximum_activation:
                function_string = "1"
            else:
                function_string = "(" + str(x) + ")/(" + str(self.maximum_activation) + ")"
        return function_string
