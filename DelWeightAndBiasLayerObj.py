"""
@author Antonio Bruce Webb(Savant)
"""
from DelWeightAndBiasObj import DelWeightAndBiasObj


"""
Class DelWeightAndBiasLayerObj: layer component for DelWeightAndBiasOrganiTensor.
"""
class DelWeightAndBiasLayerObj:

    """
    constructor for DelWeightAndBiasLayerObj.
    neural_layer: neural layer component of the neuralnetwork
    """
    def __init__(self, neural_layer):
        self.del_weight_bias_layer = []
        for perceptron_index in range(neural_layer.num_perceptrons):
            del_weight_and_bias_obj = DelWeightAndBiasObj(neural_layer.neural_layer[perceptron_index])
            self.del_weight_bias_layer.append(del_weight_and_bias_obj)

    """
    averages the del weights and biases computed during backwards prop for the perceptrons in the layer.
    """
    def average_del_weight_bias(self):
        for perceptron_index in range(len(self.del_weight_bias_layer)):
            self.del_weight_bias_layer[perceptron_index].average_del_weights_bias()

    """
    Adds a del weight and del bias calculations to the corresponding perceptron in the tensor.
    perceptron_index: index for the perceptron in the tensor.
    del_weights: array of del weights to be added to the tensor.
    del_bias: del bias to be added to the tensor.
    """
    def add_del_weight_bias_calc(self, perceptron_index, del_weights, del_bias):
        self.del_weight_bias_layer[perceptron_index].add_del_weight_bias_calc(del_weights, del_bias)

    """
    retrieves del_weight_bias_obj at the given index.
    perceptron_index: perceptron_index that maps to the del_weight_bias_obj.
    """
    def del_weight_bias_obj_at_index(self, perceptron_index):
        return self.del_weight_bias_layer[perceptron_index]

    """
    retrieves the del weight calculations fo the layer.
    """
    def del_weights(self):
        weights_list = []
        for perceptron_index in range(len(self.del_weight_bias_layer)):
            weights_list.append(self.del_weight_bias_layer[perceptron_index].del_weights())
        return weights_list

    """
    retrieves the del bias calculations for the layer.
    """
    def del_biases(self):
        biases = []
        for perceptron_index in range(len(self.del_weight_bias_layer)):
            biases.append(self.del_weight_bias_layer[perceptron_index].del_bias())
        return biases

    """
    clears the del weight and del bias calculations in the layer.
    """
    def clear(self):
        for perceptron_index in range(len(self.del_weight_bias_layer)):
            self.del_weight_bias_layer[perceptron_index].clear()