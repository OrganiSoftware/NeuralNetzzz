"""
@author Antonio Bruce Webb(Organi)
"""
from random import random


"""
class Perceptron perceptron component of the neuralnetwork.
"""
class Perceptron:

    """
    constructor for Perceptron.
    activation_funct: actiuvation function component.
    num_inputs: number of inputs.
    learning_rate: learning rate.
    hyperparam: hyperparam for the network pushes it closer to zero or further away.
    """
    def __init__(self, activation_funct, num_inputs, learning_rate, hyperparam):
        self.activation_funct = activation_funct
        self.num_inputs = num_inputs
        self.hyperparam = hyperparam
        self.learning_rate = learning_rate
        self.inputs = []
        self.weights = []
        self.bias = (2 * random()) - 1
        self.stored_weighted_sum = 0
        for input_index in range(num_inputs):
            self.weights.append((2 * random()) - 1)
        self.weights_loaded = True

    """
    retrieves the activation of the perceptron.
    """
    def activate(self):
        weighted_sum = self.calc_weighted_sum()
        return self.activation_funct.calculate(weighted_sum)

    """
    retrieves the input value at a given index.
    input_index: index of input to retrieve.
    """
    def input_at_index(self, input_index):
        return self.inputs[input_index]

    """
    retrieves the weight at a given index.
    input_index: index of weight to retrieve.
    """
    def weight_at_index(self, input_index):
        return self.weights[input_index]

    """
    adjust the weights of the perceptron.
    del_weights: del weights from the batch to adjust the weights.
    """
    def adjust_weights(self, del_weights):
        for weight_index in range(self.num_inputs):
            self.weights[weight_index] = self.weights[weight_index] - (self.learning_rate * del_weights[weight_index])

    """
    calculates the weighted sum of the perceptron.
    """
    def calc_weighted_sum(self):
        weighted_sum = 0
        for input_index in range(self.num_inputs):
            weighted_sum += self.inputs[input_index] * self.weights[input_index]
        weighted_sum += self.bias
        self.stored_weighted_sum = weighted_sum
        return weighted_sum

    """
    returns the number of inputs for the perceptron.
    """
    def number_of_inputs(self):
        return self.num_inputs

    """
    adjust the bias of the perceptron.
    del_bias: del bias from the batch to adjust the bias.
    """
    def adjust_bias(self, del_bias):
        self.bias = self.bias - (self.learning_rate * del_bias)

    """
    loads in a given input set
    inputs: set of inputs to load in.
    """
    def load_inputs(self, inputs):
        self.inputs = inputs

    """
    loads in given weights and bias.
    weights: weight to be loaded in.
    bias: bias to be loaded in.
    """
    def load_weights_bias(self, weights, bias):
        self.weights = weights
        self.bias = bias
    """
    calculates the partial of the mse cost function with respect to the weight and bias.
    weight_index: index of the weight to calc partial of.
    deriving_bias: bool if you are calculating the del bias or not.
    del_cost: del of cost function with respect to the activation.
    """
    def comp_partial_for_mse_cost(self, weight_index, deriving_bias, del_cost):
        if not deriving_bias:
            return ((self.inputs[weight_index] * self.activation_funct.comp_derivative(self.stored_weighted_sum, del_cost) * del_cost)
                    + self.hyperparam * self.weights[weight_index])
        else:
            return self.activation_funct.comp_derivative(self.stored_weighted_sum, del_cost) * del_cost

    """
    calculates the partial of the mse cost function with respect to the activation for hidden layers.
    weight_index: index of the weight to calc partial of.
    del_cost: del of cost function with respect to the activation of the layer l+1.
    """
    def calc_del_c_not_del_activation(self, weight_index, del_cost):
        return self.weights[weight_index] * self.activation_funct.comp_derivative(self.stored_weighted_sum, del_cost) * del_cost

