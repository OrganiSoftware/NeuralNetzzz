"""
@author Antonio Bruce Webb(Organi)
"""
from DataSet import DataSet


def convert(x, y, rejected_outputs):
    data_set = DataSet(255, 0)
    for image in range(len(x)):
        expected_output = int(y[image])
        inputs = []
        for row in range(len(x[image])):
            for column in range(len(x[image][row])):
                inputs.append(float(x[image][row][column]))
        data_set.add_state(inputs, expected_output, rejected_outputs)
    return data_set



