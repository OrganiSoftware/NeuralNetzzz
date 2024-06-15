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

    def train(self, epoch, batch_sizes):
        count = 0
        training_state_loaded = False
        status_string = ""
        tick_count = 0
        total_iterations = epoch * batch_sizes
        print(str(int((count / total_iterations) * 100) % 100) + "%: " + str(status_string))
        for interation in range(epoch):
            batch = []
            batch_start_index = int(random() * len(self.training_set.expected_outputs))
            self.shuffle_training_dataset()
            for training_state in range(batch_sizes):
                index = (batch_start_index + training_state) % len(self.training_set.expected_outputs)
                if len(self.training_set.inputs[index]) > 0:
                    batch.append(index)
                    self.neural_net.load_inputs(self.training_set.inputs[index])
                    if not training_state_loaded:
                        self.del_weight_bias_organi_tensor = DelWeightAndBiasOrganiTensor(self.neural_net)
                        training_state_loaded = True
                    self.comp_partial_of_w_of_cost(
                        self.neural_net.ideal_activations_for_prediction(self.training_set.expected_outputs[index],
                                                                     self.training_set.rejected_outputs[index]))
                count += 1
                if int((count / total_iterations) * 100) % 100 > tick_count:
                    for tick in range((int((count / total_iterations) * 100) % 100) - tick_count):
                        status_string += "#"
                    tick_count += (int((count / total_iterations) * 100) % 100) - tick_count
                    print(str(int((count / total_iterations) * 100) % 100) + "%: " + str(status_string))
            self.del_weight_bias_organi_tensor.average_del_weight_biases()
            self.neural_net.adjust_weights_biases(self.del_weight_bias_organi_tensor)
            self.del_weight_bias_organi_tensor.clear()
        return self.neural_net


    @staticmethod
    def is_in_batch(index, batch):
        index_in_batch = False
        for state_index in batch:
            if state_index == index:
                index_in_batch = True
        return index_in_batch

    def shuffle_training_dataset(self):
        for training_state_index in range(len(self.training_set.expected_outputs)):
            index = int((len(self.training_set.expected_outputs) - 1) * random())
            input_exchanger = self.training_set.inputs[training_state_index]
            expected_output_exchanger = self.training_set.expected_outputs[training_state_index]
            rejected_output_exchanger = self.training_set.rejected_outputs[training_state_index]
            self.training_set.inputs[training_state_index] = self.training_set.inputs[index]
            self.training_set.expected_outputs[training_state_index] = self.training_set.expected_outputs[index]
            self.training_set.rejected_outputs[training_state_index] = self.training_set.rejected_outputs[index]
            self.training_set.inputs[index] = input_exchanger
            self.training_set.expected_outputs[index] = expected_output_exchanger
            self.training_set.rejected_outputs[index] = rejected_output_exchanger

    def comp_partial_of_w_of_cost(self, ideal_activations):
        del_costs = []
        del_costs_matrix = []
        for output_index in range(len(ideal_activations)):
            del_costs.append(2 * (self.neural_net.neural_net[len(self.neural_net.neural_net) - 1].neural_layer[output_index].activate() - ideal_activations[output_index]))
            print(del_costs[output_index])
        for layer_index in range(len(self.neural_net.neural_net)):
            index = (len(self.neural_net.neural_net) - (layer_index + 1))
            temp_del_costs_matrix = []
            for perceptron_index in range(len(self.neural_net.neural_net[index].neural_layer)):
                if perceptron_index == 0:
                    for weight in self.neural_net.neural_net[index].neural_layer[perceptron_index].weights:
                        temp_del_costs_matrix.append(0.0)
                del_weights = []
                del_bias = 0.0
                if layer_index == 0:
                    for weight_index in range(len(self.neural_net.neural_net[index].neural_layer[perceptron_index].weights)):
                        del_weights.append(self.neural_net.neural_net[index].neural_layer[
                                               perceptron_index].comp_partial_for_mse_cost(weight_index, False, del_costs[perceptron_index]))
                        temp_del_costs_matrix[weight_index] += self.neural_net.neural_net[index].neural_layer[
                                                                perceptron_index].calc_del_c_not_del_activation(weight_index, del_costs[perceptron_index])
                    del_bias = self.neural_net.neural_net[index].neural_layer[
                                    perceptron_index].comp_partial_for_mse_cost(None, True, del_costs[perceptron_index])
                    self.del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(index, perceptron_index, del_weights,
                                                                                    del_bias)
                else:
                    for weight_index in range(len(self.neural_net.neural_net[index].neural_layer[perceptron_index].weights)):
                        if weight_index == 0:
                            del_bias += self.neural_net.neural_net[index].neural_layer[
                                            perceptron_index].comp_partial_for_mse_cost(None, True,
                                            del_costs_matrix[perceptron_index]/(len(self.neural_net.neural_net[index + 1].neural_layer)))
                        del_weights.append(self.neural_net.neural_net[index].neural_layer[perceptron_index
                                                                     ].comp_partial_for_mse_cost(weight_index, False, del_costs_matrix[perceptron_index]/
                                                                                                 (len(self.neural_net.neural_net[index + 1].neural_layer))))
                        temp_del_costs_matrix[weight_index] += self.neural_net.neural_net[index].neural_layer[
                                                  perceptron_index].calc_del_c_not_del_activation(weight_index,
                                                  del_costs_matrix[perceptron_index]/(len(self.neural_net.neural_net[index + 1].neural_layer)))
                    self.del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(index, perceptron_index,
                                                                                    del_weights,
                                                                                    del_bias)
            del_costs_matrix = temp_del_costs_matrix

