"""
@author Antonio Bruce Webb(Organi)
Class DelWeightAndBiasOrganiTensor: rank 3 tensor with non-conformal magnitudes for a given array in a dimension.
stores the del weights and del biases providing a one to one mapping to the weights in the neuralnetwork.
"""
from DelWeightAndBiasLayerObj import DelWeightAndBiasLayerObj


class DelWeightAndBiasOrganiTensor:

    """
    constructor for DelWeightAndBiasOrganiTensor
    neural_network: constructed neuralnetwork that you want the DelWeightAndBiasOrganiTensor to provide a mapping too.
    """
    def __init__(self, neural_net):
        self.del_weight_and_bias_network = []
        for neural_layer_index in range(neural_net.num_neural_layers()):
            del_weight_and_bias_layer_obj = DelWeightAndBiasLayerObj(neural_net.layer_at_index(neural_layer_index))
            self.del_weight_and_bias_network.append(del_weight_and_bias_layer_obj)

    """
    averages all the del weights and biases in the tensor based on how many where added in the batch.
    """
    def average_del_weight_biases(self):
        for layer_index in range(len(self.del_weight_and_bias_network)):
            self.del_weight_and_bias_network[layer_index].average_del_weight_bias()

    """
    adds del weights and del bias to the tensor for a given perceptron.
    layer_index: layer index of the neuralnetwork with which the perceptron lives.
    perceptron_index: index of the perceptron in the layer.
    del_weights: array of del weights for that were computed in backwards prop.
    del_bias: del bias computed in backwards prop.
    """
    def add_del_weight_and_bias_calc(self, layer_index, perceptron_index, del_weights, del_bias):
        self.del_weight_and_bias_network[layer_index].add_del_weight_bias_calc(perceptron_index, del_weights, del_bias)

    """
    retrieves the a multi-dimensional array containing all of the del weights and biases for a given layer 
    of the network for a given batch.
    """
    def del_weight_bias_obj_layer_at_index(self, layer_index):
        return self.del_weight_and_bias_network[layer_index]

    """
    retrieves a multi-dimensional array of  
    """
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

    def clear(self):
        for layer_index in range(len(self.del_weight_and_bias_network)):
            self.del_weight_and_bias_network[layer_index].clear()

