"""
@author Antonio Bruce Webb(Organi)
"""
import tensorflow as tf
from DataSet import DataSet
from NeuralNetwork import NeuralNetwork
from MSEOptimizer import MSEOptimizer
from SigmoidalActivationFunction import SigmoidalActivationFuction
def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_data_set = convert(x_train, y_train)
    train_data_set.store_in_json("/run/media/organi/Work/mnist_train.json")
    test_data_set = convert(x_test, y_test)
    test_data_set.store_in_json("/run/media/organi/Work/mnist_test.json")
    sigmoid = SigmoidalActivationFuction()
    output_translation_table = []
    for index in range(10):
        output_translation_table.append(index)
    neural_net = NeuralNetwork(output_translation_table, len(train_data_set.inputs[0]), sigmoid, 0.1)
    neural_net.add_hidden_layers(20,10)
    neural_net.add_hidden_layers(10,5)
    neural_net.add_hidden_layers(20,10)
    neural_net.is_constructed()
    mse_optimizer = MSEOptimizer(neural_net, train_data_set)
    neural_net = mse_optimizer.train(1, 2000)
    count = 0
    for inputs in range(len(test_data_set.inputs)):
        predicted_output = neural_net.predict_output(test_data_set.inputs[inputs])
        print(str(predicted_output))
        print(str(train_data_set.expected_outputs[inputs]))
        if predicted_output == train_data_set.expected_outputs[inputs]:
            count += 1
        print(str((count/(inputs + 1))*100))


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