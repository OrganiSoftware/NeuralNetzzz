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

    def predict_output(self, input):

    def layer_at_index(self, layer_index):

    def load_weights(self):

    def adjust_weights(self):

    def adjust_biases(self):


