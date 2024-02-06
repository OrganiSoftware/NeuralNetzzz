"""
@author Antonio Bruce Webb(Organi)
"""
from NeuralLayer import NeuralLayer


class NeuralNetwork:
    def __init__(self, size_of_output_layer, size_of_input_layer, activation_function, learning_rate, output_translation_table):
        self.neural_net = []
        self.size_of_output_layer = size_of_output_layer
        self.output_translation_table = output_translation_table
        self.size_of_input_layer = size_of_input_layer
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.constructed = False

    def add_hidden_layers(self, num_layers, size_of_layers):
        if not self.constructed:
            for layer_index in range(num_layers):
                if self.neural_net is None:
                    num_inputs = self.size_of_input_layer
                else:
                    num_inputs = len(self.neural_net[len(self.neural_net) - 1])
                layer = NeuralLayer(num_inputs, size_of_layers, self.activation_function, self.learning_rate)
                self.neural_net.append(layer)

    def constructed(self):
        if not self.constructed:
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
        predicted_output = self.output_translation_table[predicted_output_index]
        return predicted_output

    def layer_at_index(self, layer_index):
        return self.neural_net[layer_index]

    def num_neural_layers(self):
        return len(self.neural_net)

    """
    def load_weights(self, neural_memory_json):
    """

    def adjust_weights_biases(self, del_weight_bias_network):
        if self.constructed:
            for layer_index in range(len(self.neural_net)):
                self.neural_net[layer_index].adjust_weights_biases(del_weight_bias_network[layer_index])

