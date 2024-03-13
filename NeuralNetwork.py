"""
@author Antonio Bruce Webb(Organi)
"""
from NeuralLayer import NeuralLayer
from DualNumber import DualNumber
import json

class NeuralNetwork:
    def __init__(self, output_translation_table, size_of_input_layer, activation_function, learning_rate):
        self.neural_net = []
        self.size_of_output_layer = len(output_translation_table)
        self.output_translation_table = output_translation_table
        self.size_of_input_layer = size_of_input_layer
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.constructed = False

    def add_input_layer(self, num_perceptrons):
        layer = NeuralLayer(self.size_of_input_layer, num_perceptrons, self.activation_function, self.learning_rate)
        self.neural_net.append(layer)

    def add_hidden_layers(self, num_layers, size_of_layers):
        if not self.constructed:
            for layer_index in range(num_layers):
                num_inputs = len(self.neural_net[len(self.neural_net) - 1].neural_layer)
                layer = NeuralLayer(num_inputs, size_of_layers, self.activation_function, self.learning_rate)
                self.neural_net.append(layer)

    def is_constructed(self):
        if not self.constructed:
            self.add_hidden_layers(1, self.size_of_output_layer)
            self.constructed = True

    def predict_output(self, inputs):
        self.load_inputs(inputs)
        output_layer_activations = self.neural_net[len(self.neural_net) - 1].activations()
        predicted_output_value = 0
        predicted_output_index = 0
        for output_index in range(len(output_layer_activations)):
            print(str(output_layer_activations[output_index]))
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

    def load_weights_biases(self, path):
        with open(str(path), 'r') as jsonReader:
            json_data = json.load(jsonReader)
            for layer in range(len(json_data["DataSet"])):
                for perceptron in range(len(json_data["DataSet"][layer]["layer"])):
                    weights = json_data["DataSet"][layer]["layer"][perceptron]["weights"]
                    bias = json_data["DataSet"][layer]["layer"][perceptron]["bias"]
                    self.neural_net[layer].neural_layer[perceptron].load_weights_bias(weights, bias)
            jsonReader.close()

    @staticmethod
    def is_rejected(output, rejected_outputs):
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

    def comp_partial(self, ideal_activations, del_weight_bias_organi_tensor):
        for layer_index in range(len(self.neural_net)):
            for perceptron_index in range(len(self.neural_net[layer_index].neural_layer)):
                del_weights_dual_list = []
                del_bias_dual = DualNumber(0, 0)
                for weight_index in range(len(self.neural_net[layer_index].neural_layer[perceptron_index].weights) + 1):
                    initial_dual_layer_activations = []
                    initial_dual_bias_layer_activations = []
                    for prop_layer_index in range(len(self.neural_net) - layer_index):
                        dual_layer_activations = []
                        index = prop_layer_index + layer_index
                        for prop_perceptron_index in range(len(self.neural_net[index].neural_layer)):
                            if prop_layer_index == 0 and not weight_index == len(self.neural_net[layer_index].neural_layer[perceptron_index].weights):
                                if prop_perceptron_index == perceptron_index:
                                    initial_dual_layer_activations.append(self.neural_net[index].neural_layer[prop_perceptron_index].comp_partial(weight_index, False))
                                else:
                                    initial_dual_layer_activations.append(self.neural_net[index].neural_layer[prop_perceptron_index].comp_partial(None, False))
                            elif weight_index == len(self.neural_net[layer_index].neural_layer[perceptron_index].weights):
                                if prop_layer_index == 0:
                                    if prop_perceptron_index == perceptron_index:
                                        initial_dual_bias_layer_activations.append(
                                            self.neural_net[index].neural_layer[prop_perceptron_index].comp_partial(
                                                None, True))
                                    else:
                                        initial_dual_bias_layer_activations.append(
                                            self.neural_net[index].neural_layer[prop_perceptron_index].comp_partial(
                                                None, False))
                                else:
                                    dual_layer_activations.append(self.neural_net[index].neural_layer[
                                                                      prop_perceptron_index].comp_partial_given_inputs_dual(
                                        initial_dual_bias_layer_activations))
                            else:
                                dual_layer_activations.append(self.neural_net[index].neural_layer[prop_perceptron_index].comp_partial_given_inputs_dual(initial_dual_layer_activations))
                        if weight_index == len(self.neural_net[layer_index].neural_layer[perceptron_index].weights):
                            if prop_layer_index == 0:
                                initial_dual_bias_layer_activations = initial_dual_bias_layer_activations
                            else:
                                initial_dual_bias_layer_activations = dual_layer_activations
                        elif prop_layer_index == 0:
                            initial_dual_layer_activations = initial_dual_layer_activations
                        else:
                            initial_dual_layer_activations = dual_layer_activations
                    dual_partial = DualNumber(0, 0)
                    for output_index in range(len(ideal_activations)):
                        if weight_index == len(self.neural_net[layer_index].neural_layer[perceptron_index].weights):
                            dual_expected_activation = DualNumber(ideal_activations[output_index], 0)
                            dual_partial =  dual_partial + (initial_dual_bias_layer_activations[output_index] - dual_expected_activation) ** 2
                        else:
                            dual_expected_activation = DualNumber(ideal_activations[output_index], 0)
                            dual_partial = dual_partial + (initial_dual_layer_activations[output_index] - dual_expected_activation)**2
                    if weight_index == len(self.neural_net[layer_index].neural_layer[perceptron_index].weights):
                        del_bias_dual = dual_partial
                    else:
                        del_weights_dual_list.append(dual_partial)

                del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(layer_index, perceptron_index, del_weights_dual_list, del_bias_dual)
        return del_weight_bias_organi_tensor







