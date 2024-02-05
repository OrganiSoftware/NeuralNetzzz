"""
@authonr Antonio B.Webb(Organi)
"""


class DataSet:

    def __init__(self, max_value, min_value):
        self.inputs = [[]]
        self.expected_outputs = []

    def add_state(self, inputs, expected_output):
        self.inputs.append(inputs)
        self.expected_outputs.append(expected_output)

    def delete_state(self, input_index):
        temp_inputs = [[]]
        temp_expected_outputs = []
        for input in range(len(self.inputs)):
            if not input == input_index:
                temp_inputs.append(self.inputs[input])
                temp_expected_outputs.append(self.expected_outputs[input])

    def replace_state(self, input_index, new_inputs, new_expected):
        self.delete_state(input_index)
        self.inputs.append(new_inputs)
        self.expected_outputs.append(new_expected)

    def clear(self,):
        self.inputs = [[]]
        self.expected_outputs = []