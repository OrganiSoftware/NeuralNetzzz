"""
@author Antonio Bruce Webb(Organi);
"""
from Perceptron import Perceptron


"""
class NeuralLayer: neural layer component of the neuralnetwork.
"""
class NeuralLayer:

    """
    constructor for NeuralLayer.
    num_inputs: number inputs that are to be taken in by the perceptrons.
    activation_funct: the activation function component of the perceptrons.
    learning_rate: learning rate of the perceptrons.
    hyperparam: hyperparam for the network pushes it closer to zero or further away.
    """
    def __init__(self, num_inputs, num_perceptrons, activation_funct, learning_rate, hyperparam, rationalizer):
        self.neural_layer = []
        self.num_perceptrons = num_perceptrons
        self.rationalizer = rationalizer
        self.rational_num = self.rationalizer/self.num_perceptrons
        for perceptron_index in range(self.num_perceptrons):
            perceptron = Perceptron(activation_funct, num_inputs, learning_rate, hyperparam, self.rational_num)
            self.neural_layer.append(perceptron)


    """
    retrieves an array of the activations for the layer.
    """
    def activations(self):
        activations = []
        for perceptron_index in range(len(self.neural_layer)):
            activations.append(self.neural_layer[perceptron_index].activate())
        return activations

    """
    retrieves the perceptron component of a given index in the layer.
    """
    def perceptron_at_index(self, perceptron_index):
        return self.neural_layer[perceptron_index]

    """
    retrieves the number of perceptrons in the layer.
    """
    def num_perceptrons(self):
        return self.num_perceptrons

    """
    loads the inputs into the perceptrons in the layer.
    """
    def load_inputs(self, inputs):
        for perceptron_index in range(len(self.neural_layer)):
            self.neural_layer[perceptron_index].load_inputs(inputs)

    """
    adjust then weights and biases of the perceptrons in the layer.
    del_weight_bias_layer: del_weight_bias_layer component of the DelWeightAndBiasOrganiTensor.
    """
    def adjust_weights_biases(self, del_weight_bias_layer):
        for perceptron_index in range(self.num_perceptrons):
            self.neural_layer[perceptron_index].adjust_weights(del_weight_bias_layer.del_weight_bias_layer[perceptron_index].del_weights)
            self.neural_layer[perceptron_index].adjust_bias(del_weight_bias_layer.del_weight_bias_layer[perceptron_index].del_bias)
