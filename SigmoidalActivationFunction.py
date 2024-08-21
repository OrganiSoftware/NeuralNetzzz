"""
@author Antonio Bruce Webb(Savant)
"""
from math import e
from math import pow

"""
class SigmoidalActivationFuction: activation function component.
"""
class SigmoidalActivationFuction:

    """
    constructor for SigmoidalActivationFuction.
    """
    def __init__(self):
        self.function = "(1/(1+e^-x))"

    """
    calculates the output of the activation function.
    x: x value to derive the output from.
    """
    @staticmethod
    def calculate(x):
        return 1/(1+pow(e, -x))

    """
    retrieves the the derivative of the activation function.
    weighted_sum: the value for x being passed into the activation function.
    del_cost: the del cost of the cost function.
    """
    @staticmethod
    def comp_derivative(weighted_sum, del_cost):
        return (e**(weighted_sum * -1))/((1 + e**(weighted_sum * -1))**2)