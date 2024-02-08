"""
@author Antonio Bruce Webb(Organi)
"""
from DelWeightAndBiasObj import DelWeightAndBiasObj


class DelWeightAndBiasLayerObj:

    def __init__(self, neural_layer):
        self.del_weight_bias_layer = []
        for perceptron_index in range(neural_layer.num_perceptrons):
            del_weight_and_bias_obj = DelWeightAndBiasObj(neural_layer.neural_layer[perceptron_index])
            self.del_weight_bias_layer.append(del_weight_and_bias_obj)

    def average_del_weight_bias(self):
        for perceptron_index in range(len(self.del_weight_bias_layer)):
            self.del_weight_bias_layer[perceptron_index].average_del_weights_bias()

    def add_del_weight_bias_calc(self, perceptron_index, del_weights, del_bias):
        self.del_weight_bias_layer[perceptron_index].add_del_weight_bias_calc(del_weights, del_bias)

    def del_weight_bias_obj_at_index(self, perceptron_index):
        return self.del_weight_bias_layer[perceptron_index]

    def del_weights(self):
        weights_list = []
        for perceptron_index in range(len(self.del_weight_bias_layer)):
            weights_list.append(self.del_weight_bias_layer[perceptron_index].del_weights())
        return weights_list

    def del_biases(self):
        biases = []
        for perceptron_index in range(len(self.del_weight_bias_layer)):
            biases.append(self.del_weight_bias_layer[perceptron_index].del_bias())
        return biases
    def clear(self):
        for perceptron_index in range(len(self.del_weight_bias_layer)):
            self.del_weight_bias_layer[perceptron_index].clear()