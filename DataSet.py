"""
@authonr Antonio B.Webb(Organi)
"""


class DataSet:

    def __init__(self, max_value, min_value):
        self.inputs = [[]]
        self.expected_outputs = []
        self.rejected_outputs = []
        self.max_value = max_value
        self.min_value = min_value

    def add_state(self, inputs, expected_output, rejected_outputs):
        normalized_input_state = []
        for input_state in range(len(inputs)):
            normalized_input_state.append(inputs[input_state]/(self.max_value - self.min_value))
        self.inputs.append(normalized_input_state)
        self.expected_outputs.append(expected_output)
        self.rejected_outputs.append(rejected_outputs)

    def delete_state(self, input_index):
        temp_inputs = [[]]
        temp_expected_outputs = []
        temp_rejected_outputs = [[]]
        for input in range(len(self.inputs)):
            if not input == input_index:
                temp_inputs.append(self.inputs[input])
                temp_expected_outputs.append(self.expected_outputs[input])
                temp_rejected_outputs.append(self.rejected_outputs[input])
    def replace_state(self, input_index, new_inputs, new_expected, new_rejected):
        self.delete_state(input_index)
        self.add_state(new_inputs, new_expected, new_rejected)

    def clear(self,):
        self.inputs = [[]]
        self.expected_outputs = []