"""
@author Antonio Bruce Webb(Savant)
"""
from DelWeightAndBiasOrganiTensor import DelWeightAndBiasOrganiTensor
from random import random

"""
class MSEOptimizer: optimizer for training the neural network.
"""
class MSEOptimizer:

    """
    constructor for MSEOptimizer.
    neural_net: neuralnetwork component that is to be trained.
    training_set: the dataset component that is to be used for training.
    """
    def __init__(self, neural_net, training_set):
        self.neural_net = neural_net
        self.training_set = training_set
        self.del_weight_bias_organi_tensor = None

    """
    the training algorithm for the neuralnetwork using mini batch gradient descent.
    epoch: number of epoches
    batch_sizes: size of the batches.
    """
    def train(self, epoch, batch_sizes):
        count = 0
        training_state_loaded = False
        status_string = ""
        tick_count = 0
        total_iterations = epoch * batch_sizes
        print(str(int((count / total_iterations) * 100) % 100) + "%: " + str(status_string))
        for interation in range(epoch):
            self.shuffle_training_dataset()
            batch_start_index = int(random() * len(self.training_set.expected_outputs))
            for training_state_index in range(batch_sizes):
                index = (batch_start_index + training_state_index) % len(self.training_set.expected_outputs)
                print(index)
                if len(self.training_set.inputs[index]) > 0:
                    self.neural_net.load_inputs(self.training_set.inputs[index])
                    if not training_state_loaded:
                        self.del_weight_bias_organi_tensor = DelWeightAndBiasOrganiTensor(self.neural_net)
                        training_state_loaded = True
                    self.comp_partial_of_w_of_cost(self.neural_net.ideal_activations_for_prediction(self.training_set.expected_outputs[index],
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

    """
    shuffles the training set randomly
    """
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

    """
    calculates the del costs and del biases for a training state in the optimization.
    ideal_activations: the expected output of the neuralnetwork.
    """
    def comp_partial_of_w_of_cost(self,ideal_activations):
        del_costs = []
        del_costs_matrix = []
        for output_index in range(len(ideal_activations)):
            del_costs.append((self.neural_net.neural_net[len(self.neural_net.neural_net) - 1
                              ].neural_layer[output_index].activate() - ideal_activations[output_index]))
            print(del_costs[output_index])
        for layer_index in range(len(self.neural_net.neural_net)):
            index = (len(self.neural_net.neural_net) - (layer_index + 1))
            temp_del_costs_matrix = []
            for weight_index in range(len(self.neural_net.neural_net[index].neural_layer[0].weights)):
                temp_del_costs_matrix.append(0.0)
            for perceptron_index in range(len(self.neural_net.neural_net[index].neural_layer)):
                del_weights = []
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

                        del_weights.append(self.neural_net.neural_net[index].neural_layer[perceptron_index
                                                                     ].comp_partial_for_mse_cost(weight_index, False, del_costs_matrix[perceptron_index]))
                        temp_del_costs_matrix[weight_index] += self.neural_net.neural_net[index].neural_layer[
                                                perceptron_index].calc_del_c_not_del_activation(weight_index,
                                                                                                del_costs_matrix[perceptron_index])
                    del_bias = self.neural_net.neural_net[index].neural_layer[perceptron_index].comp_partial_for_mse_cost(None, True,
                                                                                                                          del_costs_matrix[perceptron_index])
                    self.del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(index, perceptron_index,
                                                                                    del_weights,
                                                                                    del_bias)
            del_costs_matrix = temp_del_costs_matrix



