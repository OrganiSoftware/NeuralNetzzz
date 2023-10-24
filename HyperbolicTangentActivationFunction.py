"""
@author Antonio Bruce Webb(Organi)
"""

from FunctionParser import FunctionParser


class HyperbolicTangentActivationFunction:

    def __init__(self):
        self.function = "((e^x-e^-x)/(e^x+e^-x))"

    def calcualate(self, x):
        return self.function_parser.calculate(x)

