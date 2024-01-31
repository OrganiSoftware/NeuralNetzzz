"""
@author Antonio Bruce Webb(Organi)
"""
from math import e
from math import pow

class HyperbolicTangentActivationFunction:

    def __init__(self):
        self.function = "(e^x - e^-x) / (e^x + e^-x)"
    def calculate(self, x):
        return (pow(e, x) - (pow(e, -x))) / (pow(e, x) + (pow(e, -x)))

    def function_string(self, x):
        function_string = "(((e^"+str(x)+") - (e^ - "+str(x)+")) / ((e^"+str(x)+") + (e^ - "+str(x)+")))"
        return function_string
