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
        sum = 0
        for input_index in range(len(self.inputs)):
            sum += self.inputs[input_index] * self.weights[input_index] + self.bias
        return self.activation_funct.activate(sum)

    def input_at_index(self, input_index):
        return self.inputs[input_index]

    def weight_at_index(self, input_index):
        return self.weights[input_index]

    def optimize_weights(self, del_weights):
        for weight_index in range(len(self.weights)):
            self.weights[weight_index] = self.weights[weight_index] - (self.learning_rate * del_weights[weight_index])

    def number_of_inputs(self):
        return self.num_inputs

    def optimize_bias(self, del_bias):
        self.bias = self.bias - (self.learning_rate * del_bias)

    def load_input(self, inputs):
        self.inputs = inputs





