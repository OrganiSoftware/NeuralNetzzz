"""
@author Antonio Bruce Webb(Organi)
"""

from FunctionParser import FunctionParser


class SigmoidalActivationFuction:

    def __init__(self):
        self.function = "((e^x-e^-x)/(e^x+e^-x))"
        self.function_parser = FunctionParser(self.function)

    def calcualate(self, x):
        return self.function_parser.calculate(x)

