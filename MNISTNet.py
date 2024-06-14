from DataSet import DataSet
from NeuralNetwork import NeuralNetwork
from MSEOptimizer import MSEOptimizer
from SigmoidalActivationFunction import SigmoidalActivationFuction
from LeakySquashedRELUActivationFunction import LeakySquashedRELUActivationFunction


def main():
    train_data_set = DataSet(1,0)
    test_data_set = DataSet(1,0)
    train_data_set.json_load("/run/media/jackal/Work/SoftwareProjects/NeuralNetzzz/mnist_train.json", 1)
    test_data_set.json_load("/run/media/jackal/Work/SoftwareProjects/NeuralNetzzz/mnist_test.json", 1)
    sigmoid = SigmoidalActivationFuction()
    relu = LeakySquashedRELUActivationFunction(0, 10)
    output_translation_table = []
    for index in range(10):
        output_translation_table.append(index)
    num_inputs = 0
    for train_data_state in range(len(train_data_set.inputs)):
        if not len(train_data_set.inputs[train_data_state]) < 0:
            num_inputs = len(train_data_set.inputs[train_data_state])
            break
    neural_net = NeuralNetwork(output_translation_table, num_inputs, sigmoid, 1)
    neural_net.add_input_layer(32)
    neural_net.add_hidden_layers(1, 16)
    neural_net.add_hidden_layers(1, 32)
    neural_net.is_constructed()
    mse_optimizer = MSEOptimizer(neural_net, train_data_set)
    neural_net = mse_optimizer.train(10000,64)
    neural_net.save_weights_biases("/run/media/jackal/Work/SoftwareProjects/NeuralNetzzz/weights_bias.json")
    count = 0
    for inputs in range(len(test_data_set.expected_outputs)):
        if not len(test_data_set.inputs[inputs]) == 0:
            predicted_output = neural_net.predict_output(test_data_set.inputs[inputs])
            print(str(predicted_output))
            print(str(test_data_set.expected_outputs[inputs]))
            if predicted_output == test_data_set.expected_outputs[inputs]:
                count += 1
            print(str((count/(inputs + 1))*100))


if __name__ == "__main__":
    main()