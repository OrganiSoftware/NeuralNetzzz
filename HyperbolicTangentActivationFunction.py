"""
@author Antonio Bruce Webb(Organi)
from FunctionParser import FunctionParser
"""
from math import e
from math import pow

class HyperbolicTangentActivationFunction:

    def __init__(self):
        self.function = "((e^x-e^-x)/(e^x+e^-x))"
    @staticmethod
    def calculate(self, x):
        return (pow(e, x) - (pow(e, -x))) / (pow(e, x) + (pow(e, -x)))

    def function_string(self):
        return self.function
