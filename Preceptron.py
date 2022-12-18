import math

class Preceptron:
    def __int__(self, activation_funct, num_inputs, learning_rate):
        self.activation_funct = activation_funct
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.inputs = []
        self.weights = []
        self.bias = math.random()
        self.max_input = 1
        self.min_input = 0
        self.max_weight = 1
        self.min_weight = 0
        self.max_bias = 1
        self.min_bias = 0
        for input_index in range(self.num_inputs):
            self.inputs.append(math.random())
            self.weights.append(math.random())

    def activate(self):

    def input_at_index(self):

    def weight_at_index(self):

    def optimize_weights(self):

    def optimize_bias(self):

    def load_input(self):

    def load_saved_weights(self):





