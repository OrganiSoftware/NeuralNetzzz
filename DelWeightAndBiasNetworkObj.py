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

