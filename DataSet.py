"""
@authonr Antonio B.Webb(Organi)
"""

import json


class DataSet:

    def __init__(self, max_value, min_value):
        self.inputs = [[]]
        self.expected_outputs = []
        self.rejected_outputs = [[]]
        self.max_value = max_value
        self.min_value = min_value
        self.normalized_max = min_value
        self.normalized_min = max_value

    def add_state(self, inputs, expected_output, rejected_outputs):
        normalized_input_state = []
        for input_index in range(len(inputs)):
            normalized_value = inputs[input_index] / (self.max_value - self.min_value)
            if normalized_value < self.normalized_min:
                self.normalized_min = normalized_value
            if normalized_value > self.normalized_max:
                self.normalized_max = normalized_value
            normalized_input_state.append(normalized_value)
        self.inputs.append(normalized_input_state)
        self.expected_outputs.append(expected_output)
        self.rejected_outputs.append(rejected_outputs)

    def store_in_json(self, path):
        with open(str(path), 'w', encoding="utf-8") as jsonWriter:
            array = []
            for input_state_index in range(len(self.expected_outputs)):
                array.append({"inputs": self.inputs[input_state_index],
                              "expected_output": self.expected_outputs[input_state_index],
                              "rejected_outputs": self.rejected_outputs[input_state_index],
                              "num_inputs":len(self.inputs[input_state_index]),
                              "num_states": len(self.expected_outputs),
                              "max": self.normalized_max,
                              "min": self.normalized_min})
            jsonWriter.write(json.dumps({"DataSet": array}))
            jsonWriter.close()

    def json_load(self, path, size_of_subset):
        with open(str(path), 'r') as jsonReader:
            json_data = json.load(jsonReader)
            for input_state in json_data['DataSet']:
                subset = []
                inputs = input_state['inputs']
                average = 0
                count = 0
                for input_index in range(len(inputs)):
                    if (input_index % (size_of_subset) == 0 and not input_index == 0) or size_of_subset == 1:
                        if not count == 0:
                            subset.append(average/count)
                        else:
                            subset.append(inputs[input_index])
                        count = 0
                        average = 0
                    else:
                        average += inputs[input_index]
                        count += 1
                expected_output = input_state["expected_output"]
                rejected_outputs = input_state["rejected_outputs"]
                self.max_value = input_state['max']
                self.min_value = input_state['min']
                self.add_state(subset,expected_output,rejected_outputs)

            jsonReader.close()

    def delete_state(self, input_index):
        temp_inputs = [[]]
        temp_expected_outputs = []
        temp_rejected_outputs = [[]]
        for input in range(len(self.expected_outputs)):
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
        self.rejected_outputs = [[]]