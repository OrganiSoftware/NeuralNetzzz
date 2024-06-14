"""
@author Antonio Bruce Webb(Organi)
"""
from DataSet import DataSet


def convert(x_train, y_train):
    data_set = DataSet(255, 0)
    for image in range(len(x_train)):
        expected_output = int(y_train[image])
        inputs = []
        for row in range(len(x_train[image])):
            for column in range(len(x_train[image][row])):
                inputs.append(float(x_train[image][row][column]))
        data_set.add_state(inputs, expected_output, None)
    return data_set


