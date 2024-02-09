"""
@author Antonio Bruce Webb(Organi)
"""
from NeuralLayer import NeuralLayer


class NeuralNetwork:
    def __init__(self, output_translation_table, size_of_input_layer, activation_function, learning_rate):
        self.neural_net = []
        self.size_of_output_layer = len(output_translation_table)
        self.output_translation_table = output_translation_table
        self.size_of_input_layer = size_of_input_layer
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.constructed = False

    def add_hidden_layers(self, num_layers, size_of_layers):
        if not self.constructed:
            for layer_index in range(num_layers):
                if len(self.neural_net) == 0:
                    num_inputs = self.size_of_input_layer
                else:
                    num_inputs = len(self.neural_net[len(self.neural_net) - 1].neural_layer)
                layer = NeuralLayer(num_inputs, size_of_layers, self.activation_function, self.learning_rate)
                self.neural_net.append(layer)

    def is_constructed(self):
        if not self.constructed:
            self.add_hidden_layers(1, self.size_of_output_layer)
            self.constructed = True

    def predict_output(self, inputs):
        if self.constructed:
           self.load_inputs(inputs)
        output_layer_activations = self.neural_net[len(self.neural_net) - 1].activations()
        predicted_output_value = 0
        predicted_output_index = 0
        for output_index in range(len(output_layer_activations)):
            if predicted_output_value < output_layer_activations[output_index]:
                predicted_output_value = output_layer_activations[output_index]
                predicted_output_index = output_index
        predicted_output = self.output_translation_table[predicted_output_index]
        return predicted_output

    def load_inputs(self, inputs):
        for layer_index in range(len(self.neural_net)):
            if layer_index == 0:
                self.neural_net[layer_index].load_inputs(inputs)
            else:
                activations = self.neural_net[layer_index - 1].activations()
                self.neural_net[layer_index].load_inputs(activations)


    def layer_at_index(self, layer_index):
        return self.neural_net[layer_index]

    def num_neural_layers(self):
        return len(self.neural_net)

    """
    def load_weights(self, neural_memory_json):
    """
    def ideal_activations_for_prediction(self, expected_output, rejected_outputs):
        ideal_activations = []
        for output_index in range(len(self.output_translation_table)):
            if self.output_translation_table[output_index] == expected_output:
                ideal_activations.append(1)
            elif self.is_rejected(self.output_translation_table[output_index], rejected_outputs):
                ideal_activations.append(-1)
            else:
                ideal_activations.append(0)
        return ideal_activations

    def is_rejected(self, output, rejected_outputs):
        is_rejected = False
        if rejected_outputs is not None:
            for rejected_output_index in range(len(rejected_outputs)):
                if output == rejected_outputs[rejected_output_index]:
                    is_rejected = True
        return is_rejected

    def adjust_weights_biases(self, del_weight_bias_network):
        if self.constructed:
            for layer_index in range(len(self.neural_net)):
                self.neural_net[layer_index].adjust_weights_biases(del_weight_bias_network.del_weight_and_bias_network[layer_index])

