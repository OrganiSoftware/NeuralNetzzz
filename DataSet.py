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
        for input in range(len(inputs)):
            normalized_value = inputs[input]/(self.max_value - self.min_value)
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
            for input_state_index in range(len(self.expected_outputs)):
                jsonWriter.write(json.dumps({"inputs": self.inputs[input_state_index],
                                             "expected_output": self.expected_outputs[input_state_index],
                                             "rejected_outputs": self.rejected_outputs[input_state_index],
                                             "num_inputs": len(self.inputs),
                                             "num_states"+str(input_state_index): len(self.inputs[input_state_index]),
                                             "max": self.normalized_max,
                                             "min": self.normalized_min}))
            jsonWriter.close()

    def json_load(self, path):
        with open(str(path), 'r') as jsonReader:
            json_data = json.loads(json.dumps(jsonReader.read()))
            print(json_data)
            self.max_value = json_data[0]['max']
            self.min_value = json_data[0]['min']
            for input_state_index in range(len(json_data)):
                inputs = json_data[input_state_index]["inputs"]
                expected_output = json_data[input_state_index]["expected_output"]
                rejected_outputs = json_data[input_state_index]["rejected_outputs"]
                self.add_state(inputs,expected_output,rejected_outputs)
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