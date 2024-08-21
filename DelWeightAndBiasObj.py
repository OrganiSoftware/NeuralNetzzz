"""
@author Antonio Bruce Webb(Savant)
"""


"""
class DelWeightAndBiasObj: del weight and bias component for the DelWeightAndBiasOrganiTensor
"""
class DelWeightAndBiasObj:

    """
    constructor for DelWeightAndBiasObj.
    perceptron: perceptron component to create a one to one mapping of.
    """
    def __init__(self, perceptron):
        self.del_weights = []
        self.del_bias = 0.0
        self.num_weight_calcs = 0
        self.averaged = False
        for weight_index in range(perceptron.number_of_inputs()):
            self.del_weights.append(0.0)

    """
    adds the del weights and del bias that were derived from backwards prop.
    del_weights: array of del_weights to be added
    del_bias: del_bias to be added.
    """
    def add_del_weight_bias_calc(self, del_weights, del_bias):
        if not self.averaged:
            for weight_index in range(len(self.del_weights)):
                self.del_weights[weight_index] = self.del_weights[weight_index] + del_weights[weight_index]
            self.del_bias = self.del_bias + del_bias
            self.num_weight_calcs += 1

    """
    averages the del weights and biases based on the number of del weights and del biases that were added in
    the batch.
    """
    def average_del_weights_bias(self):
        if not self.averaged and not self.num_weight_calcs == 0:
            for weight_index in range(len(self.del_weights)):
                self.del_weights[weight_index] = self.del_weights[weight_index] / self.num_weight_calcs
            self.del_bias = self.del_bias / self.num_weight_calcs
            self.averaged = True

    """
    retrieves an array of del weights only if the del weights have been averaged.
    """
    def del_weights(self):
        if self.averaged:
            return self.del_weights
        return None

    """
    clears del weights and biases.
    """
    def clear(self):
        self.del_bias = 0.0
        for weight_index in range(len(self.del_weights)):
            self.del_weights[weight_index] = 0.0
        self.averaged = False
        self.num_weight_calcs = 0

    """
    retrieves the del bias only if it has been averaged.
    """
    def del_bias(self):
        if self.averaged:
            return self.del_bias
        return None
