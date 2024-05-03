"""
@author Antonio Bruce Webb(Organi)
"""
from DualNumber import DualNumber

class DelWeightAndBiasObj:

    def __init__(self, perceptron):
        self.del_weights = []
        self.del_bias = 0.0
        self.num_weight_calcs = 0
        self.averaged = False
        for weight_index in range(perceptron.number_of_inputs()):
            self.del_weights.append(0.0)

    def add_del_weight_bias_calc(self, del_weights, del_bias):
        if not self.averaged:
            for weight_index in range(len(self.del_weights)):
                self.del_weights[weight_index] = self.del_weights[weight_index] + del_weights[weight_index]
            self.del_bias = self.del_bias + del_bias
            self.num_weight_calcs += 1

    def average_del_weights_bias(self):
        if not self.averaged and not self.num_weight_calcs == 0:
            for weight_index in range(len(self.del_weights)):
                self.del_weights[weight_index] = self.del_weights[weight_index] / self.num_weight_calcs
            self.del_bias = self.del_bias / self.num_weight_calcs
            self.averaged = True

    def del_weights(self):
        if self.averaged:
            return self.del_weights
        return None

    def clear(self):
        self.del_bias = 0.0
        for weight_index in range(len(self.del_weights)):
            self.del_weights[weight_index] = 0.0
        self.averaged = False
        self.num_weight_calcs = 0

    def del_bias(self):
        if self.averaged:
            return self.del_bias
        return None
