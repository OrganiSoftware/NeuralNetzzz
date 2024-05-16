"""
@author Antonio Bruce Webb(Organi)
"""
from DataSet import DataSet
from NeuralNetwork import NeuralNetwork
from MSEOptimizer import MSEOptimizer
from SigmoidalActivationFunction import SigmoidalActivationFuction
from LeakySquashedRELUActivationFunction import LeakySquashedRELUActivationFunction
def main():
    train_data_set = DataSet(1,0)
    test_data_set = DataSet(1,0)
    train_data_set.json_load("/run/media/organi/Work/mnist_train.json", 32)
    test_data_set.json_load("/run/media/organi/Work/mnist_test.json", 32)
    sigmoid = SigmoidalActivationFuction()
    relu = LeakySquashedRELUActivationFunction(0, 10)
    output_translation_table = []
    for index in range(10):
        output_translation_table.append(index)
    num_inputs = 0
    for train_data_state in range(len(train_data_set.inputs)):
        if not len(train_data_set.inputs[train_data_state]) == 0:
            num_inputs = len(train_data_set.inputs[train_data_state])
            break
    neural_net = NeuralNetwork(output_translation_table, num_inputs, sigmoid, .1)
    neural_net.add_input_layer(16)
    neural_net.add_hidden_layers(1,16)
    neural_net.is_constructed()
    mse_optimizer = MSEOptimizer(neural_net, train_data_set)
    neural_net = mse_optimizer.train(100,32)
    neural_net.save_weights_biases("/run/media/organi/Work/weights_bias.json")
    count = 0
    for inputs in range(len(test_data_set.expected_outputs)):
        if not len(test_data_set.inputs[inputs]) == 0:
            neural_net.load_inputs(test_data_set.inputs[inputs])
            predicted_output = neural_net.predict_output(test_data_set.inputs[inputs])
            #print(str(predicted_output))
            #print(str(test_data_set.expected_outputs[inputs]))
            if predicted_output == test_data_set.expected_outputs[inputs]:
                count += 1
            #rint(str((count/(inputs + 1))*100))


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


if __name__ == "__main__":
    main()