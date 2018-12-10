import report
import neural_network_generator as aptnet
import activation_functions

# This is a test for XOR
a = aptnet.NeuralNet(2, [2], 1)
a . set_activation_f(activation_functions.SIGMOID)
samples_input = [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]]
samples_output = [[0],
                  [1],
                  [1],
                  [0]]
a.set_training_samples(samples_input, samples_output)
a.train()