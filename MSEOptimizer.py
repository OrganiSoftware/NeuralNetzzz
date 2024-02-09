from DualNumber import DualNumber
from DelWeightAndBiasOrganiTensor import DelWeightAndBiasOrganiTensor
class MSEOptimizer:

    def __init__(self, neural_net, training_set):
        self.neural_net = neural_net
        self.training_set = training_set
        self.del_weight_bias_organi_tensor = None

    def calc_total_cost_of_all_states(self):
        sum_cost = 0
        for training_state in range(len(self.training_set)):
            self.neural_net.load_inputs(self.training_set.inputs[training_state])
            network_activations = []
            predicted_outputs = []
            cost_of_state = 0
            for neural_layer in range(len(self.neural_net.neural_net)):
                layer_activations = self.neural_net.neural_net[neural_layer].activations()
                network_activations.append(layer_activations)
                predicted_outputs = network_activations[len(network_activations) - 1]
            for perceptron in range(len(predicted_outputs)):
                cost_of_state += (predicted_outputs[perceptron] - self.training_set.expected_outputs[perceptron])**2
            cost_of_state = cost_of_state * (1 / len(predicted_outputs))
            sum_cost += cost_of_state

    def train(self, number_of_iterations, batch_sizes):
        count = 0
        for interation in range(number_of_iterations):
            for training_state in range(len(self.training_set.expected_outputs)):
                self.neural_net.load_inputs(self.training_set.inputs[training_state])
                if training_state == 0:
                    self.del_weight_bias_organi_tensor = DelWeightAndBiasOrganiTensor(self.neural_net)
                self.comp_network_delta_organi_tensor(self.training_set.inputs[training_state],
                                                      self.neural_net.ideal_activations_for_prediction(self.training_set.expected_outputs[training_state],
                                                                                                       self.training_set.rejected_outputs[training_state]))
                if (count % batch_sizes) == (batch_sizes - 1):
                    self.del_weight_bias_organi_tensor.average_del_weight_biases()
                    self.neural_net.adjust_weights_biases(self.del_weight_bias_organi_tensor)
                    self.del_weight_bias_organi_tensor.clear()
                count += 1
        return self.neural_net

    def comp_network_delta_organi_tensor(self, training_state_inputs, training_state_expected_outputs):
        self.neural_net.load_inputs(training_state_inputs)
        for output_perceptron in range(len(self.neural_net.neural_net[len(self.neural_net.neural_net) - 1].neural_layer)):
            del_weights = []
            m = DualNumber(len(self.neural_net.neural_net[len(self.neural_net.neural_net) - 1].neural_layer), 0)
            dual_expected = DualNumber(training_state_expected_outputs[output_perceptron], 0)
            bias_dual_partial = self.neural_net.neural_net[len(self.neural_net.neural_net) - 1].neural_layer[output_perceptron].comp_partial(None, True)
            del_bias_dual = (bias_dual_partial - dual_expected)*(bias_dual_partial - dual_expected)
            del_bias_dual = del_bias_dual / m
            del_bias = del_bias_dual.d
            for weight_index in range(len(self.neural_net.neural_net[len(self.neural_net.neural_net) - 1].neural_layer[output_perceptron].weights)):
                weight_dual_partial = self.neural_net.neural_net[len(self.neural_net.neural_net) - 1].neural_layer[output_perceptron].comp_partial(weight_index, False)
                del_weight_dual = (weight_dual_partial - dual_expected)*(weight_dual_partial - dual_expected)
                del_weight_dual = del_weight_dual / m
                del_weight = del_weight_dual.d
                del_weights.append(del_weight)
            self.del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(len(self.neural_net.neural_net) - 1, output_perceptron, del_weights,
                                                                            del_bias)
            self.propagate_through_hidden_layers(dual_expected, m, training_state_inputs)

    def propagate_through_hidden_layers(self, dual_expected, m, training_state_inputs):
        self.neural_net.load_inputs(training_state_inputs)
        for neural_layer in range(len(self.neural_net.neural_net) - 2):
            index = len(self.neural_net.neural_net) - (neural_layer + 2)
            for perceptron in range(len(self.neural_net.neural_net[index].neural_layer)):
                del_weights = []
                bias_dual_partial = self.neural_net.neural_net[index].neural_layer[perceptron].comp_partial(None, True)
                del_bias_dual = (bias_dual_partial - dual_expected)*(bias_dual_partial - dual_expected)
                del_bias_dual = del_bias_dual / m
                del_bias = del_bias_dual.d
                for weight_index in range(len(self.neural_net.neural_net[index].neural_layer[perceptron].weights)):
                    weight_dual_partial = self.neural_net.neural_net[index].neural_layer[perceptron].comp_partial(
                        weight_index, False)
                    del_weight_dual = (weight_dual_partial - dual_expected)*(weight_dual_partial - dual_expected)
                    del_weight_dual = del_weight_dual / m
                    del_weights.append(del_weight_dual.d)
                self.del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(index, perceptron, del_weights,
                                                                                del_bias)

