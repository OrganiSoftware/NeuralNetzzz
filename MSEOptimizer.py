"""
@author Antonio Bruce Webb(Organi)
"""
from DualNumber import DualNumber
from DelWeightAndBiasOrganiTensor import DelWeightAndBiasOrganiTensor
from random import random
from threading import Thread
import resource

class MSEOptimizer:

    def __init__(self, neural_net, training_set):
        self.neural_net = neural_net
        self.training_set = training_set
        self.del_weight_bias_organi_tensor = None

    def train(self, epoch, batch_sizes, threads):
        resc = resource.RLIMIT_CPU
        soft,hard = resource.getrlimit(resc)
        resource.setrlimit(resc,(soft, resource.RLIM_INFINITY))
        resc = resource.RLIMIT_MEMLOCK
        soft, hard = resource.getrlimit(resc)
        resource.setrlimit(resc, (soft, resource.RLIM_INFINITY))
        resc = resource.RLIMIT_CORE
        soft, hard = resource.getrlimit(resc)
        resource.setrlimit(resc, (soft, resource.RLIM_INFINITY))
        resc = resource.RUSAGE_THREAD
        soft, hard = resource.getrlimit(resc)
        resource.setrlimit(resc, (soft, resource.RLIM_INFINITY))
        count = 0
        training_state_loaded = False
        status_string = ""
        tick_count = 0
        print(str(int((0 / epoch) * 100) % 100) + "%: " + str(status_string))
        for iteration in range(epoch):
            self.shuffle_training_dataset()
            batch_start_index = int(random() * len(self.training_set.expected_outputs))
            batch_count = 0
            thread_count = 0
            thread_array = []
            threads_started = False
            are_threads_running = False
            if threads <= batch_sizes:
                threads_2_gen = threads
            else:
                threads_2_gen = batch_sizes
            while (batch_count < batch_sizes):
                if ((batch_sizes - batch_count) < threads_2_gen):
                    threads_2_gen = batch_sizes - batch_count
                for thread_index in range(threads_2_gen):
                    index = (batch_start_index + thread_count) % len(self.training_set.expected_outputs)
                    if len(self.training_set.inputs[index]) > 0:
                        self.neural_net.load_inputs(self.training_set.inputs[index])
                        if not training_state_loaded:
                            self.del_weight_bias_organi_tensor = DelWeightAndBiasOrganiTensor(self.neural_net)
                            training_state_loaded = True
                        t = Thread(target=self.comp_partial_of_w_of_cost, args=(self.neural_net.ideal_activations_for_prediction(
                                                                                    self.training_set.expected_outputs[
                                                                                        index],
                                                                                    self.training_set.rejected_outputs[
                                                                                        index]), self.neural_net))
                        thread_array.append(t)
                    thread_count += 1
                    count += 1
                if not threads_started:
                    for thread in thread_array:
                        thread.start()
                    threads_started = True
                    are_threads_running = True
                else:
                    while(are_threads_running):
                        are_threads_running = False
                        for thread_index in range(len(thread_array)):
                            print(thread_array[thread_index].is_alive())
                            if thread_array[thread_index].is_alive():
                                are_threads_running = True
                        if not are_threads_running:
                            batch_count += len(thread_array)
                            thread_array = []
                            thread_count = 0
                            threads_started = False
            if int((iteration / epoch) * 100) % 100 > tick_count:
                for tick in range((int((iteration / epoch) * 100) % 100) - tick_count):
                    status_string += "#"
                tick_count = (int((iteration / epoch) * 100) % 100)
                print(str(int((iteration / epoch) * 100) % 100) + "%: " + str(status_string))
            self.del_weight_bias_organi_tensor.average_del_weight_biases()
            self.neural_net.adjust_weights_biases(self.del_weight_bias_organi_tensor)
            self.del_weight_bias_organi_tensor.clear()
        return self.neural_net

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

    def comp_partial_of_w_of_cost(self,ideal_activations, neural_net):
        del_costs = []
        del_costs_matrix = []
        for output_index in range(len(ideal_activations)):
            del_costs.append(2 * (neural_net.neural_net[len(neural_net.neural_net) - 1
                                  ].neural_layer[output_index].activate() - ideal_activations[output_index]))
            print(del_costs[output_index])
        for layer_index in range(len(neural_net.neural_net)):
            index = (len(neural_net.neural_net) - (layer_index + 1))
            temp_del_costs_matrix = []
            for weight_index in range(len(neural_net.neural_net[index].neural_layer[0].weights)):
                temp_del_costs_matrix.append(0.0)
            for perceptron_index in range(len(neural_net.neural_net[index].neural_layer)):
                del_weights = []
                if layer_index == 0:
                    for weight_index in range(len(neural_net.neural_net[index].neural_layer[perceptron_index].weights)):
                        del_weights.append(neural_net.neural_net[index].neural_layer[
                                               perceptron_index].comp_partial_for_mse_cost(weight_index, False, del_costs[perceptron_index]))
                        temp_del_costs_matrix[weight_index] += neural_net.neural_net[index].neural_layer[
                                                                perceptron_index].calc_del_c_not_del_activation(weight_index, del_costs[perceptron_index])
                    del_bias = neural_net.neural_net[index].neural_layer[
                                    perceptron_index].comp_partial_for_mse_cost(None, True, del_costs[perceptron_index])
                    self.del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(index, perceptron_index, del_weights,
                                                                                    del_bias)
                else:
                    for weight_index in range(len(neural_net.neural_net[index].neural_layer[perceptron_index].weights)):

                        del_weights.append(neural_net.neural_net[index].neural_layer[perceptron_index
                                                                     ].comp_partial_for_mse_cost(weight_index, False, del_costs_matrix[perceptron_index]))
                        temp_del_costs_matrix[weight_index] += neural_net.neural_net[index].neural_layer[
                                                  perceptron_index].calc_del_c_not_del_activation(weight_index,
                                                                                                  del_costs_matrix[perceptron_index])
                    del_bias = neural_net.neural_net[index].neural_layer[perceptron_index].comp_partial_for_mse_cost(None, True,
                                                                         del_costs_matrix[perceptron_index])
                    self.del_weight_bias_organi_tensor.add_del_weight_and_bias_calc(index, perceptron_index,
                                                                                    del_weights,
                                                                                    del_bias)
            del_costs_matrix = temp_del_costs_matrix



