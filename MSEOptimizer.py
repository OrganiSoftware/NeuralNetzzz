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
        training_state_loaded = False
        status_string = ""
        tick_count = 0
        total_iterations = int(number_of_iterations * len(self.training_set.expected_outputs))
        print(str(int((count / total_iterations) * 100) % 100) + "%: " + str(status_string))
        for training_state in range(total_iterations):
            state = training_state % len(self.training_set.expected_outputs)
            if not len(self.training_set.inputs[state]) == 0:
                self.neural_net.load_inputs(self.training_set.inputs[state])
                if not training_state_loaded:
                    self.del_weight_bias_organi_tensor = DelWeightAndBiasOrganiTensor(self.neural_net)
                    training_state_loaded = True
                self.comp_partial(self.neural_net.ideal_activations_for_prediction(self.training_set.expected_outputs[state],
                                                                              self.training_set.rejected_outputs[state]))

                if (count % batch_sizes) == (batch_sizes - 1):
                    self.del_weight_bias_organi_tensor.average_del_weight_biases()
                    self.neural_net.adjust_weights_biases(self.del_weight_bias_organi_tensor)
                    self.del_weight_bias_organi_tensor.clear()
                count += 1
                if int((count/total_iterations) * 100) % 100 > tick_count:
                    tick_count += 1
                    status_string += "#"
                    print(str(int((count/total_iterations) * 100) % 100)+"%: "+str(status_string))
        return self.neural_net

    def comp_partial(self, ideal_activations):
        for layer_index in range(len(self.neural_net)):
            for perceptron_index in range(len(self.neural_net.neural_net[layer_index].neural_layer)):
                del_weights_dual_list = []
                del_bias_dual = DualNumber(0, 0)
                for weight_index in range(len(self.neural_net.neural_net[layer_index].neural_layer[perceptron_index].weights) + 1):
                    initial_dual_layer_activations = []
                    initial_dual_bias_layer_activations = []
                    for prop_layer_index in range(len(self.neural_net.neural_net) - layer_index):
                        dual_layer_activations = []
                        index = prop_layer_index + layer_index
                        for prop_perceptron_index in range(len(self.neural_net.neural_net[index].neural_layer)):
                            if prop_layer_index == 0 and not weight_index == len(
                                    self.neural_net.neural_net[layer_index].neural_layer[perceptron_index].weights):
                                if prop_perceptron_index == perceptron_index:
                                    initial_dual_layer_activations.append(
                                        self.neural_net.neural_net[index].neural_layer[prop_perceptron_index].comp_partial(
                                            weight_index, False))
                                else:
                                    initial_dual_layer_activations.append(
                                        self.neural_net.neural_net[index].neural_layer[prop_perceptron_index].comp_partial(None,
                                                                                                                False))
                            elif weight_index == len(
                                    self.neural_net.neural_net[layer_index].neural_layer[perceptron_index].weights):
                                if prop_layer_index == 0:
                                    if prop_perceptron_index == perceptron_index:
                                        initial_dual_bias_layer_activations.append(
                                            self.neural_net.neural_net[index].neural_layer[prop_perceptron_index].comp_partial(
                                                None, True))
                                    else:
                                        initial_dual_bias_layer_activations.append(
                                            self.neural_net.neural_net[index].neural_layer[prop_perceptron_index].comp_partial(
                                                None, False))
                                else:
                                    dual_layer_activations.append(self.neural_net.neural_net[index].neural_layer[
                                        prop_perceptron_index].comp_partial_given_inputs_dual(
                                        initial_dual_bias_layer_activations))
                            else:
                                dual_layer_activations.append(self.neural_net.neural_net[index].neural_layer[
                                                                  prop_perceptron_index].comp_partial_given_inputs_dual(
                                    initial_dual_layer_activations))
                        if weight_index == len(self.neural_net.neural_net[layer_index].neural_layer[perceptron_index].weights):
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
                        if weight_index == len(self.neural_net.neural_net[layer_index].neural_layer[perceptron_index].weights):
                            dual_expected_activation = DualNumber(ideal_activations[output_index], 0)
                            dual_partial = dual_partial + (initial_dual_bias_layer_activations[
                                                               output_index] - dual_expected_activation) ** 2
                        else:
                            dual_expected_activation = DualNumber(ideal_activations[output_index], 0)
                            dual_partial = dual_partial + (
                                        initial_dual_layer_activations[output_index] - dual_expected_activation) ** 2
                    if weight_index == len(self.neural_net.neural_net[layer_index].neural_layer[perceptron_index].weights):
                        del_bias_dual = dual_partial
                    else:
                        del_weights_dual_list.append(dual_partial)

                self.del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(layer_index, perceptron_index,
                                                                                del_weights_dual_list, del_bias_dual)