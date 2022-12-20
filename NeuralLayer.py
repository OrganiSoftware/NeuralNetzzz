"""
@author Antonio Bruce Webb(Jackal)
"""
from Preceptron import Preceptron


class NeuralLayer:
    def __init__(self, num_inputs, num_preceptrons, activation_funct, learning_rate):
        self.neural_layer = []
        self.num_preceptrons = num_preceptrons
        for preceptron_index in range(self.num_preceptrons):
            preceptron = Preceptron(activation_funct, num_inputs, learning_rate)
            self.neural_layer.append(preceptron)

    def activations(self):
        activations = []
        for preceptron_index in range(len(self.neural_layer)):
            activations.append(self.neural_layer[preceptron_index].activate())
        return activations

    def preceptron_at_index(self, preceptron_index):
        return self.neural_layer[preceptron_index]

    def num_preceptrons(self):
        return self.num_preceptrons()

    def load_inputs(self, inputs):
        for preceptron_index in range(self.num_preceptrons):
            self.neural_layer[preceptron_index].load_inputs(inputs)

    #def load_weights(self):

    def adjust_weights_biases(self, del_weight_bias_layer):
        for preceptron_index in range(self.num_preceptrons):
            self.neural_layer[preceptron_index].adjust_weights(del_weight_bias_layer[preceptron_index])
            self.neural_layer[preceptron_index].adjust_bias(del_weight_bias_layer[preceptron_index].del_bias())
