from DualNumber import DualNumber
class MSEOptimizer:

    def _init__(self, neural_net, training_set, del_weight_net_obj):
        self.neural_net = neural_net
        self.training_set = training_set
        self.del_weight_net_obj = del_weight_net_obj

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
                cost_of_state += (predicted_outputs[perceptron] - self.training_set[perceptron])**2
            cost_of_state = cost_of_state * (1 / len(predicted_outputs))
            sum_cost += cost_of_state

    def train(self, number_of_iterations, batch_sizes):
        for interation in range(number_of_iterations):
            for training_state in range(len(self.training_set)):
                self.neural_net.load_inputs(self.training_set.inputs[training_state])
                for output_index in range(len(self.neural_net.neural_net[len(self.neural_net.neural_net) - 1])):
                    for layer_index in range(len(self.neural_net.neural_net) - 1):
                        index = (len(self.neural_net.neural_net) - 2) - layer_index
                        for perceptron_index in range(len(self.neural_net.neural_net[index].neural_layer)):
                            del_weights = 0
                            del_bais = self.neural_net.neural_net[index].neural_layer[perceptron_index].comp_partial(0, True)


    def comp_partial_list(self, perceptron):
        one = DualNumber(1, 0)
