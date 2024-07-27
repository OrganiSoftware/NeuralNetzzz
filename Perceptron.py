"""
@author Antonio Bruce Webb(Organi)
"""
from random import random

from pfp.fields import false

from DualNumber import DualNumber


class Perceptron:

    def __init__(self, activation_funct, num_inputs, learning_rate, hyperparam, activation_function_type):
        self.activation_funct = activation_funct
        self.num_inputs = num_inputs
        self.hyperparam = hyperparam
        self.learning_rate = learning_rate
        self.inputs = []
        self.weights = []
        self.bias = 2 * random() - 1
        self.stored_weighted_sum = 0
        for input_index in range(num_inputs):
            self.weights.append((2 * random()) - 1)
        self.weights_loaded = True

    def activate(self):
        weighted_sum = self.calc_weighted_sum()
        return self.activation_funct.calculate(weighted_sum)

    def input_at_index(self, input_index):
        return self.inputs[input_index]

    def weight_at_index(self, input_index):
        return self.weights[input_index]

    def adjust_weights(self, del_weights):
        for weight_index in range(self.num_inputs):
            self.weights[weight_index] = self.weights[weight_index] - (self.learning_rate * del_weights.del_weights[weight_index])

    def calc_weighted_sum(self):
        weighted_sum = 0
        for input_index in range(self.num_inputs):
            weighted_sum += self.inputs[input_index] * self.weights[input_index]
        weighted_sum += self.bias
        self.stored_weighted_sum = weighted_sum
        return weighted_sum

    def number_of_inputs(self):
        return self.num_inputs

    def adjust_bias(self, del_bias):
        self.bias = self.bias - (self.learning_rate * del_bias)

    def load_inputs(self, inputs):
        self.inputs = inputs

    def load_weights_bias(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def comp_partial(self, weight_index, deriving_bias, deriving_x):
        weighted_sum = 0
        dual_weighted_sum = None
        for input_index in range(self.num_inputs):
            if weight_index == input_index and not deriving_bias and weight_index and not deriving_x:
                weight_dual_num = DualNumber(self.weights[input_index], 1)
                input_dual_num = DualNumber(self.inputs[input_index], 0)
                bias_dual_num = DualNumber(self.bias, 0)
            elif deriving_x and weight_index == input_index:
                weight_dual_num = DualNumber(self.weights[input_index], 0)
                input_dual_num = DualNumber(self.inputs[input_index], 1)
                bias_dual_num = DualNumber(self.bias, 0)
            elif deriving_bias:
                weight_dual_num = DualNumber(self.weights[input_index], 0)
                input_dual_num = DualNumber(self.inputs[input_index], 0)
                bias_dual_num = DualNumber(self.bias, 1)
            else:
                weight_dual_num = DualNumber(self.weights[input_index], 0)
                input_dual_num = DualNumber(self.inputs[input_index], 0)
                bias_dual_num = DualNumber(self.bias, 0)
            if dual_weighted_sum is None:
                dual_weighted_sum = (input_dual_num * weight_dual_num) + bias_dual_num
            else:
                dual_weighted_sum = dual_weighted_sum + ((input_dual_num * weight_dual_num) + bias_dual_num)
            weighted_sum += self.inputs[input_index] * self.weights[input_index] + self.bias
        return self.activation_funct.comp_partial(weighted_sum, dual_weighted_sum)

    def comp_partial_given_inputs_dual(self, dual_inputs):
        weighted_sum = 0
        dual_weighted_sum = None
        for input_index in range(len(dual_inputs)):
            weight_dual_num = DualNumber(self.weights[input_index], 0)
            input_dual_num = dual_inputs[input_index]
            if dual_weighted_sum is None:
                dual_weighted_sum = (input_dual_num * weight_dual_num) + DualNumber(self.bias, 0)
            else:
                dual_weighted_sum = dual_weighted_sum + ((input_dual_num * weight_dual_num) + DualNumber(self.bias, 0))
            weighted_sum += self.inputs[input_index] * self.weights[input_index] + self.bias
        return self.activation_funct.comp_partial(weighted_sum, dual_weighted_sum)

    def comp_partial_for_mse_cost(self, weight_index, deriving_bias, del_cost):
        if not deriving_bias:
            return ((self.inputs[weight_index] * self.activation_funct.comp_derivative(self.stored_weighted_sum, del_cost) * del_cost)
                    + self.hyperparam * self.weights[weight_index])
        else:
            return self.activation_funct.comp_derivative(self.stored_weighted_sum, del_cost) * del_cost

    def calc_del_c_not_del_activation(self, weight_index, del_cost):
        return self.weights[weight_index] * self.activation_funct.comp_derivative(self.stored_weighted_sum, del_cost) * del_cost

