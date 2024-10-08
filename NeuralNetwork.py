"""
@author Antonio Bruce Webb(Savant)
"""
from NeuralLayer import NeuralLayer
import json


"""
class NeuralNetwork: neuralnetwork component.
"""
class NeuralNetwork:

    """
    constructor for NeuralNetwork.
    output_translation_table: mapping of outputs to the outputs of the network.
    size_of_input_layer: number of inputs passed into the input layer.
    activation_funct: the activation function component of the perceptrons.
    learning_rate: learning rate of the perceptrons.
    hyperparam: hyperparam for the network pushes it closer to zero or further away.
    """
    def __init__(self, output_translation_table, size_of_input_layer, activation_function, learning_rate, hyperparam, rational_num):
        self.neural_net = []
        self.size_of_output_layer = len(output_translation_table)
        self.hyperparam = hyperparam
        self.output_translation_table = output_translation_table
        self.size_of_input_layer = size_of_input_layer
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.constructed = False
        self.rational_num = rational_num
        self.rationalizer = self.rational_num * self.size_of_input_layer

    """
    adds an input layer to the neuralnetwork.
    num_perceptrons: number  of perceptrons in the input layer.
    """
    def add_input_layer(self, num_perceptrons):
        layer = NeuralLayer(self.size_of_input_layer, num_perceptrons, self.activation_function, self.learning_rate,
                            self.hyperparam, self.rationalizer)
        self.neural_net.append(layer)

    """
    adds hidden layers to the neuralnetwork.
    num_layers: number of hidden layers to be added.
    size_of_layers: size of the hidden layers to be added.
    """
    def add_hidden_layers(self, num_layers, size_of_layers):
        for layer_index in range(num_layers):
            num_inputs = len(self.neural_net[len(self.neural_net) - 1].neural_layer)
            if self.constructed:
                layer = NeuralLayer(num_inputs, size_of_layers, self.activation_function, self.learning_rate,
                                        self.hyperparam, self.size_of_output_layer)
            else:
                layer = NeuralLayer(num_inputs, size_of_layers, self.activation_function, self.learning_rate,
                                        self.hyperparam, self.rationalizer)
            self.neural_net.append(layer)

    """
    checks if neuralnetwork is constructed and adds an output layer if it is not. 
    """
    def is_constructed(self):
        if not self.constructed:
            self.constructed = True
            self.add_hidden_layers(1, self.size_of_output_layer)

    """
    predicts an output based on a given input set.
    inputs: array of values that the network needs to make a prediction from.
    """
    def predict_output(self, inputs):
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

    """
    forward propogates the inputs through the network.
    inputs: array of values that the network needs to forward propogate.
    """
    def load_inputs(self, inputs):
        for layer_index in range(len(self.neural_net)):
            if layer_index == 0:
                self.neural_net[layer_index].load_inputs(inputs)
            else:
                activations = self.neural_net[layer_index - 1].activations()
                self.neural_net[layer_index].load_inputs(activations)

    """
    retrieves the neural layer at a given index.
    layer_index: index of the layer that is to be retrieved.
    """
    def layer_at_index(self, layer_index):
        return self.neural_net[layer_index]

    """
    retrieves the number of layers in the network.
    """
    def num_neural_layers(self):
        return len(self.neural_net)

    """
    retrieves the expected activations of the network.
    expected_output: expected output of the network
    rejected_outputs: rejected outputs of the network.
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

    """
    saves weights and biases in a formatted json.
    path: path to save the json to.
    """
    def save_weights_biases(self, path):
        with open(str(path), 'w', encoding="utf-8") as jsonWriter:
            data = []
            for layer_index in range(len(self.neural_net)):
                weights = []
                for perceptron_index in range(len(self.neural_net[layer_index].neural_layer)):
                    weights.append({"weights": self.neural_net[layer_index].neural_layer[perceptron_index].weights,
                                    "bias": self.neural_net[layer_index].neural_layer[perceptron_index].bias})
                data.append({"layer": weights})
            jsonWriter.write(json.dumps({"DataSet": data}))
            jsonWriter.close()

    """
    loads weights and biases from a formatted json.
    path: path to json file to load.
    """
    def load_weights_biases(self, path):
        with open(str(path), 'r') as jsonReader:
            json_data = json.load(jsonReader)
            for layer in range(len(json_data["DataSet"])):
                for perceptron in range(len(json_data["DataSet"][layer]["layer"])):
                    weights = json_data["DataSet"][layer]["layer"][perceptron]["weights"]
                    bias = json_data["DataSet"][layer]["layer"][perceptron]["bias"]
                    self.neural_net[layer].neural_layer[perceptron].load_weights_bias(weights, bias)
            jsonReader.close()

    """
    checks if the output is in the list of rejected outputs.
    output: the output that is to be checked.
    rejected_outputs: list of rejected outputs.
    """
    @staticmethod
    def is_rejected(output, rejected_outputs):
        is_rejected = False
        if rejected_outputs is not None:
            for rejected_output_index in range(len(rejected_outputs)):
                if output == rejected_outputs[rejected_output_index]:
                    is_rejected = True
        return is_rejected

    """
    adjust the weights and biases of the network.
    del_weight_bias_network: DelWeightAndBiasOrganiTensor object.
    """
    def adjust_weights_biases(self, del_weight_bias_network):
        if self.constructed:
            for layer_index in range(len(self.neural_net)):
                self.neural_net[layer_index].adjust_weights_biases(del_weight_bias_network.del_weight_and_bias_network[layer_index])


    def clear_inputs(self):
        for layer_index in range(len(self.neural_net)):
            for perceptron in self.neural_net[layer_index].neural_layer:
                perceptron.clear_inputs()





