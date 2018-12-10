# We use these functions to see what is going on in our neural network
# Most of these functions are created for debugging purposes
import neural_network_generator as net


def print_weights(neural_network: net.NeuralNet):
    for x in neural_network.weights:
        print("----------------------------------------\n")
        print(x)

def print_biases(neural_network: net.NeuralNet):
    for x in neural_network.biases:
        print("----------------------------------------\n")
        print(x)


def print_input_values(neural_network: net.NeuralNet):
    if neural_network.input_values is None:
        print("Error: plz set the inputs values")
    else:
        print(neural_network.input_values)


def print_network(neural_network: net.NeuralNet):
    print(neural_network.input_values)
    for x in neural_network.hidden_values:
        print(x)
    print(neural_network.output_values)
