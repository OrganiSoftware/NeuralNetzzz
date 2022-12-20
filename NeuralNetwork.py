"""
@author Antonio Bruce Webb(Jackal)
"""
from NeuralLayer import NeuralLayer


class NeuralNetwork:
    def __init__(self, size_of_output_layer, size_of_input_layer, activation_function, learning_rate):
        self.neural_net = []
        self.size_of_output_layer = size_of_output_layer
        self.size_of_input_layer = size_of_input_layer
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.constructed = False

    def add_hidden_layers(self, num_layers, size_of_layers):
        if not self.constructed:
            for layer_index in range(num_layers):
                num_inputs = 0
                if self.neural_net is None:
                    num_inputs = self.size_of_input_layer
                else:
                    num_inputs = len(self.neural_net[len(self.neural_net) - 1])
                layer = NeuralLayer(num_inputs, size_of_layers, self.activation_function, self.learning_rate)
                self.neural_net.append(layer)

    def constructed(self):
        self.add_hidden_layers(1, self.size_of_output_layer)
        self.constructed = True

    def predict_output(self, inputs):
        if self.constructed:
            for layer_index in range(len(self.neural_net)):
                if layer_index == 0:
                    self.neural_net[layer_index].load_inputs(inputs)
                else:
                    activations = self.neural_net[layer_index - 1].activations()
                    self.neural_net[layer_index].load_inputs(activations)
        output_layer_activations = self.neural_net[len(self.neural_net) - 1].activations()
        predicted_output_value = 0
        predicted_output_index = 0
        for output_index in range(len(output_layer_activations)):
            if predicted_output_value < output_layer_activations[output_index]:
                predicted_output_value = output_layer_activations[output_index]
                predicted_output_index = output_index
        return predicted_output_index

    def layer_at_index(self, layer_index):
        return self.neural_net[layer_index]

    def load_weights(self):

    def adjust_weights(self):

    def adjust_biases(self):


