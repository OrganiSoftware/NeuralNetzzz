"""
@author Antonio Bruce Webb(Organi)
"""
from DualNumber import DualNumber
from DelWeightAndBiasOrganiTensor import DelWeightAndBiasOrganiTensor
from random import random
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

    def train(self, epoch, batch_sizes):
        count = 0
        training_state_loaded = False
        status_string = ""
        tick_count = 0
        total_iterations = epoch * batch_sizes
        for interation in range(epoch):
            for training_state in range(batch_sizes):

        print(str(int((count / total_iterations) * 100) % 100) + "%: " + str(status_string))
        for training_state in range(total_iterations):
            state = training_state % len(self.training_set.expected_outputs)
            if not len(self.training_set.inputs[state]) == 0:
                self.neural_net.load_inputs(self.training_set.inputs[state])
                if not training_state_loaded:
                    self.del_weight_bias_organi_tensor = DelWeightAndBiasOrganiTensor(self.neural_net)
                    training_state_loaded = True
                self.comp_partial_of_w_of_cost(self.neural_net.ideal_activations_for_prediction(self.training_set.expected_outputs[state],
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

    def comp_partial_of_w_of_cost(self, ideal_activations):
        del_costs = []
        del_costs_matrix = [[]]
        for output_index in range(len(ideal_activations)):
            del_costs.append(2 * (self.neural_net.neural_net[len(self.neural_net.neural_net) - 1].neural_layer[output_index].activate() - ideal_activations[output_index]))
        for layer_index in range(len(self.neural_net.neural_net)):
            index = (len(self.neural_net.neural_net) - (layer_index + 1))
            temp_del_costs_matrix = [[]]
            for perceptron_index in range(len(self.neural_net.neural_net[index].neural_layer)):
                temp_del_costs = []
                if perceptron_index == 0:
                    for weight in self.neural_net.neural_net[index].neural_layer[perceptron_index].weights:
                        temp_del_costs_matrix.append([])
                del_weights = []
                del_bias = 0.0
                if layer_index == 0:
                    for weight_index in range(len(self.neural_net.neural_net[index].neural_layer[perceptron_index].weights)):
                        del_weights.append(self.neural_net.neural_net[index].neural_layer[
                                               perceptron_index].comp_partial_for_mse_cost(weight_index, False, del_costs[perceptron_index]))
                        temp_del_costs.append(self.neural_net.neural_net[index].neural_layer[
                                                  perceptron_index].calc_del_c_not_del_activation(weight_index, del_costs[perceptron_index]))
                    del_bias = self.neural_net.neural_net[index].neural_layer[
                        perceptron_index].comp_partial_for_mse_cost(None, True, del_costs[perceptron_index])
                    self.del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(index, perceptron_index, del_weights,
                                                                                    del_bias)
                    for temp_cost_index in range(len(temp_del_costs)):
                        temp_del_costs_matrix[temp_cost_index].append(temp_del_costs[temp_cost_index])
                else:
                    for weight_index in range(len(self.neural_net.neural_net[index].neural_layer[perceptron_index].weights)):
                        temp_del_weight = 0.0
                        for del_cost in del_costs_matrix[perceptron_index]:
                            if weight_index == 0:
                                del_bias += self.neural_net.neural_net[index].neural_layer[
                                                                            perceptron_index].comp_partial_for_mse_cost(None, True, del_cost)
                            temp_del_weight += self.neural_net.neural_net[index].neural_layer[
                                               perceptron_index].comp_partial_for_mse_cost(weight_index, False, del_cost)
                            temp_del_costs_matrix[weight_index].append(self.neural_net.neural_net[index].neural_layer[
                                                  perceptron_index].calc_del_c_not_del_activation(weight_index, del_cost))
                        temp_del_weight = temp_del_weight/len(del_costs_matrix[perceptron_index])
                        if weight_index == 0:
                            del_bias = del_bias/len(del_costs_matrix[perceptron_index])
                        del_weights.append(temp_del_weight)
                    self.del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(index, perceptron_index,
                                                                                    del_weights,
                                                                                    del_bias)
            del_costs_matrix = temp_del_costs_matrix