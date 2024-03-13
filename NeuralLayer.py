"""
@author Antonio Bruce Webb(Organi);
"""
from Perceptron import Perceptron


class NeuralLayer:
    def __init__(self, num_inputs, num_perceptrons, activation_funct, learning_rate):
        self.neural_layer = []
        self.num_perceptrons = num_perceptrons
        for perceptron_index in range(self.num_perceptrons):
            perceptron = Perceptron(activation_funct, num_inputs, learning_rate)
            self.neural_layer.append(perceptron)

    def activations(self):
        activations = []
        for perceptron_index in range(len(self.neural_layer)):
            activations.append(self.neural_layer[perceptron_index].activate())
        return activations

    def perceptron_at_index(self, perceptron_index):
        return self.neural_layer[perceptron_index]

    def num_perceptrons(self):
        return self.num_perceptrons()

    def load_inputs(self, inputs):
        for perceptron_index in range(self.num_perceptrons):
            self.neural_layer[perceptron_index].load_inputs(inputs)

    #def load_weights(self):

    def adjust_weights_biases(self, del_weight_bias_layer):
        for perceptron_index in range(self.num_perceptrons):
            self.neural_layer[perceptron_index].adjust_weights(del_weight_bias_layer.del_weight_bias_layer[perceptron_index])
            self.neural_layer[perceptron_index].adjust_bias(del_weight_bias_layer.del_weight_bias_layer[perceptron_index].del_bias)
