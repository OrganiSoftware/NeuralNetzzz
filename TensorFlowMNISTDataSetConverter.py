"""
@author Antonio Bruce Webb(Organi)
"""
from DataSet import DataSet

"""
converter for tensorflow datasets.
x: x_train or x_test of the dataset.
y: y_train or y_test of the dataset.
rejected_outputs: multi-dimensional array of rejected outputs for the dataset. 
"""
def convert(x, y, rejected_outputs):
    data_set = DataSet(255, 0)
    for image in range(len(x)):
        expected_output = int(y[image - 1])
        inputs = []
        x[image] = x[image].reshape(28,28)
        for row in range(len(x[image])):
            for column in range(len(x[image][row])):
                inputs.append(x[image][row][column])
        if rejected_outputs is None:
            data_set.add_state(inputs, expected_output, None)
        else:
            data_set.add_state(inputs, expected_output, rejected_outputs[image])
    return data_set



