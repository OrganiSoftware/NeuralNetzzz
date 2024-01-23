
class MSECostFunction:

    def _init__(self, neural_net, training_set):
        self.neural_net = neural_net
        self.training_set = training_set

    def calc_total_cost_of_all_states(self):
        sum_cost = 0
        for training_state in self.training_set:
            self.neural_net.load_inputs(training_state.inputs)
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




