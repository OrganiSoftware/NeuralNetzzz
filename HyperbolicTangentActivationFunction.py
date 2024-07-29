"""
@author Antonio Bruce Webb(Organi)
"""
from math import e
from math import pow

"""
class HyperbolicTangentActivationFunction: activation function class.
"""
class HyperbolicTangentActivationFunction:

    """
    constructor for HyperbolicTangentActivationFunction
    """
    def __init__(self):
        self.function = "(e^x - e^-x) / (e^x + e^-x)"

    """
    activation function.
    x: the x value that you are passing into the activation function.
    """
    @staticmethod
    def calculate(x):
        return (pow(e, x) - (pow(e, -x))) / (pow(e, x) + (pow(e, -x)))

    """
    retrieves the the derivative of the activation function.
    weighted_sum: the value for x being passed into the activation function.
    del_cost: the del cost of the cost function.
    """
    def comp_derivative(self, weighted_sum, del_cost):
        return 1 - ((self.calculate(weighted_sum))**2)
