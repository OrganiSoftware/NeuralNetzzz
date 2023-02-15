"""
@author Antonio Bruce Webb(Organi)
"""
from DelWeightAndBiasLayerObj import DelWeightAndBiasLayerObj


class DelWeightAndBiasNetworkObj:

    def __init__(self, neural_net):
        self.del_weight_and_bias_network = []
        for neural_layer_index in range(neural_net.num_neural_layers()):
            del_weight_and_bias_layer_obj = DelWeightAndBiasLayerObj(neural_net.layer_at_index())
            self.del_weight_and_bias_network.append(del_weight_and_bias_layer_obj)

    def average_del_weight_biases(self):
        for layer_index in range(len(self.del_weight_and_bias_network)):
            self.del_weight_and_bias_network[layer_index].average_del_weight_bias()

    def add_del_weight_and_bias_calc(self, layer_index, preceptron_index, del_weights, del_bias):
        self.del_weight_and_bias_network[layer_index].add_del_weight_bias_calc(preceptron_index, del_weights, del_bias)

    def del_weight_bias_obj_layer_at_index(self, layer_index):
        return self.del_weight_and_bias_network[layer_index]

    def del_weights(self):
        weights_list = []
        for layer_index in range(len(self.del_weight_and_bias_network)):
            weights_list.append(self.del_weight_and_bias_network[layer_index].del_weights())
        return weights_list

    def del_biases(self):
        del_biases = []
        for layer_index in range(len(self.del_weight_and_bias_network)):
            del_biases.append(self.del_weight_and_bias_network[layer_index].del_biases())
        return del_biases

    
