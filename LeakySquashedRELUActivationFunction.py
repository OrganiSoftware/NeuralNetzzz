"""
@author Antonio Bruce Webb(Organi)
"""


"""
class LeakySquashedRELUActivationFunction: activation function class.
"""
class LeakySquashedRELUActivationFunction:

    """
    constructor for LeakySquashedRELUActivationFunction.
    alpha: the leaky part of the activation function this is the slope when x < 0.
    maximum_activation: max activation of the relu.
    minimum_activation: min activation of the relu.
    """
    def __init__(self, alpha, maximum_activation, minimum_activation):
        self.alpha = alpha
        self.maximum_activation = maximum_activation
        self.minimum_activation = minimum_activation

    """
    activation function.
    x: the x value the is to be passed into the activation function.
    """
    def calculate(self, x):
        if x < 0:
            if x * self.alpha < self.minimum_activation:
                resultant = -1
            else:
                if self.alpha <= 0 or self.minimum_activation == 0:
                    if x * self.alpha > self.maximum_activation:
                        resultant = 1
                    else:
                        resultant = (x * self.alpha)/self.maximum_activation
                else:
                    resultant = (x * self.alpha)/-self.minimum_activation
        else:
            if x > self.maximum_activation:
                resultant = 1
            else:
                resultant = x / self.maximum_activation
        return resultant

    """
    retrieves the derivative of the activation function.
    x: the value for x being passed into the activation function.
    del_cost: the del cost of the cost function.
    """
    def comp_derivative(self, x, del_cost):
        derivative = 0.0
        if (0 >= x > self.minimum_activation) or (x * self.alpha <= self.minimum_activation and del_cost < 0) or (x < 0 and self.alpha < 0):
            if (self.alpha == 0 and del_cost < 0) or (x == 0 and del_cost < 0):
                derivative = 1 / self.maximum_activation
            else:
                if self.alpha <= 0 or self.minimum_activation == 0:
                    if ((self.alpha * x) <= self.maximum_activation) and del_cost < 0:
                        derivative = 0
                    else:
                        derivative = self.alpha/self.maximum_activation
                else:
                    derivative = self.alpha/-self.minimum_activation
        if (0 < x  < self.maximum_activation) or (x >= self.maximum_activation and del_cost > 0):
            derivative = 1 / self.maximum_activation
        return derivative

