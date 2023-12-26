
class MSECostFunction:

    def _init__(self, neural_net, training_set):
        self.neural_net = neural_net
        self.training_set = training_set

    def calc_total_cost(self):
        sum_cost = 0
        for training_state in self.training_set:
            self.neural_net.load_inputs(training_state.inputs)
            network_activations = []
            for neural_layer in range(len(self.neural_net.neural_net)):
                layer_activations = self.neural_net.neural_net[neural_layer].activations()
                network_activations.append(layer_activations)
                predicted_outputs = network_activations[len(network_activations) - 1]
            for preceptron in range(len(self.neural_net.neural_net[len(self.neural_net.neural_net) - 1].neural_layer)):
                
                sum_cost +=  (1/2*len(preceptron.weights))



