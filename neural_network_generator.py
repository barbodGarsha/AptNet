import activation_functions as af
import numpy as np
import error_handler as e


class NeuralNet:

    def __init__(self, input_layer, hidden_layer, output_layer):
        #  Inputs layer | Hidden layers | Output layer
        #               | H1-1    H2-1  |
        #       X1      | H1-2    H2-2  |    Z1
        #       X2      | H1-3    H2-3  |    Z2
        #       X3      | H1-4    H2-4  |    Z3
        #               | H1-5    H2-5  |
        #
        # input_layer is an integer and shows the size of the input layer.
        # hidden_layer is an array of integers and shows the size of the hidden layer.
        # len[hidden_layer] shows how many hidden layer we have.
        # hidden_layer[n] shows the size of the hidden layer number n
        # output_layer is an integer and shows the size of the output layer.
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        # we will automatically create arrays for our weights and biases with the information above
        self.weights = []
        self.biases = []

        # these arrays will keep the value of each neuron
        self.input_values = []
        self.hidden_values = []
        self.output_values = []

        # this array will keep the activation function that we will use for each layer
        # we use one activation function for the whole network but maybe in the future we use different functions for
        # each layer
        self.activation_f = []

        # Now we generate the weights and biases with random values based on the shape of the neural network
        self.weights = generate_weights(self.input_layer, self.hidden_layer, self.output_layer)
        self.biases = generate_biases(self.weights)

        # It contain the data-set for the training function
        self.training_samples = [TrainingData]
        # These arrays will help us to calculate the backpropagation easier
        self.training_layers = []
        self.training_weights = []
        self.training_biases = []
        self.training_pure_values = []

    def feedforward(self):
        # This list will keep the value of every neuron before we use the activation function
        self.training_pure_values = []
        self.training_pure_values.append(self.input_values)
        # right now Cleaning these array makes no difference
        # but for more complicated neural networks it might be needed
        self.hidden_values = []
        self.output_values = []
        if len(self.hidden_layer) != 0:
            y = add_biases(np.dot(self.input_values, self.weights[0]), self.biases[0])
            self.training_pure_values.append(af.sigmoid_p_array(y))
            x = next_layer(self.input_values, self.weights[0], self.biases[0], self.activation_f[0])
            self.hidden_values.append(x)
            if len(self.hidden_layer) > 1:
                for i in range(len(self.hidden_layer) - 1):
                    y = add_biases(np.dot(self.hidden_values[i], self.weights[i + 1]), self.biases[i + 1])
                    self.training_pure_values.append(af.sigmoid_p_array(y))
                    x = next_layer(self.hidden_values[i], self.weights[i + 1],
                                   self.biases[i + 1], self.activation_f[i + 1])
                    self.hidden_values.append(x)
                else:
                    y = add_biases(np.dot(self.hidden_values[i + 1], self.weights[i + 2]), self.biases[i + 2])
                    self.training_pure_values.append(af.sigmoid_p_array(y))
                    self.output_values = next_layer(self.hidden_values[i + 1], self.weights[i + 2],
                                                    self.biases[i + 2], self.activation_f[i + 2])
            else:
                y = add_biases(np.dot(self.hidden_values[0], self.weights[1]), self.biases[1])
                self.training_pure_values.append(af.sigmoid_p_array(y))
                self.output_values = next_layer(self.hidden_values[0], self.weights[1],
                                                self.biases[1], self.activation_f[1])
        else:
            y = add_biases(np.dot(self.input_values, self.weights[0]), self.biases[0])
            self.training_pure_values.append(af.sigmoid_p_array(y))
            self.output_values = next_layer(self.input_values, self.weights[0], self.biases[0], self.activation_f[0])

    def train(self):
        # Input the samples --> feedforward --> calculate the cost --> backpropagation
        cost_d = 0.0
        for y in range(200000):
            for x in self.training_samples:
                self.set_input_values(x.input)
                self.feedforward()
                cost_d = calculate_err_d(x.output[0], self.output_values[0])
                self.backpropagation(cost_d)
                if y % 2000 == 0:
                    print(calculate_err(x.output[0], self.output_values[0]))
            if y % 2000 == 0:
                print(".................")
        print("END")
        for x in self.training_samples:
            self.set_input_values(x.input)
            self.feedforward()
            print(self.output_values)
    # This fill the input layer
    def set_input_values(self, input_values):

        if len(input_values) == self.input_layer:
            self.input_values = input_values
        else:
            e.error(e.ERROR_ARRAY_SIZES, "input_values size does not match with self.input_values")

    # We will use these training samples to train the neural network
    def set_training_samples(self, input_values, output_values):
        # Note: I wanted to check the whole data-set here but I will decide later where to put it

        if len(input_values) == len(output_values) and len(input_values) > 0 and len(output_values) > 0:
            for x in range(len(input_values)):
                if len(input_values[x]) != self.input_layer or len(output_values[x]) != self.output_layer:
                    e.error(e.ERROR_ARRAY_SIZES, "data-set and the neural network are not matched")
            self.training_samples = []
            for x in range(len(input_values)):
                self.training_samples.append(TrainingData(input_values[x], output_values[x]))
        else:
            e.error(e.ERROR_ARRAY_SIZES, "Some info in training data-set is missing")

    # This function will ready the training network
    def create_training_net(self):

        # TO DO: we should add training_activation_f for later
        self.training_layers = []
        self.training_biases = []
        self.training_weights = []

        self.training_layers.append(self.output_values)
        for x in reversed(self.hidden_values):
            self.training_layers.append(x)
        self.training_layers.append(self.input_values)

        for x in reversed(self.weights):
            new_list = np.zeros((x.shape[1], x.shape[0]))
            for i in range(x.shape[1]):
                for j in range(x.shape[0]):
                    new_list[i, j] = x[j, i]
            self.training_weights.append(new_list)

        for x in reversed(self.biases):
            self.training_biases.append(x)

        self.training_pure_values.reverse()

    def db_pre_layer(self, layer):
        new_list = np.zeros((len(self.training_layers[layer]), len(self.training_layers[layer + 1])))
        for x in range(len(self.training_layers[layer])):
            for y in range(len(self.training_layers[layer + 1])):
                new_list[x, y] = self.training_pure_values[layer][x] * self.training_weights[layer][x, y]
        return new_list

    def db_pre_weights(self, layer):
        new_list = np.zeros((len(self.training_layers[layer]), len(self.training_layers[layer + 1])))
        for x in range(len(self.training_layers[layer])):
            for y in range(len(self.training_layers[layer + 1])):
                new_list[x, y] = self.training_pure_values[layer][x] * self.training_layers[layer + 1][y]
        return new_list

    def d_cost_weight(self, layer, cost_d):
        if layer == 0:
            return np.dot(cost_d, self.db_pre_weights(0))
        z = np.dot(cost_d, self.db_pre_layer(0))
        for x in range(1, layer):
            z = np.dot(z, self.db_pre_layer(x))
        new_list = np.zeros((len(self.training_layers[layer]), len(self.training_layers[layer + 1])))
        for i in range(z.shape[1]):
            for y in range(len(self.training_layers[layer + 1])):
                new_list[i, y] = z[0, i] * self.db_pre_weights(layer)[i, y]
        return new_list

    def d_cost_bias(self, layer, cost_d):
        if layer == 0:
            return np.dot(cost_d, [1])
        z = np.dot(cost_d, self.db_pre_layer(0))
        for x in range(1, layer):
            z = np.dot(z, self.db_pre_layer(x))
        new_list = self.training_biases[layer]
        for i in range(z.shape[1]):
            for y in range(len(new_list)):
                new_list[y] = z[0, i]
        return new_list

    def update(self):
        new_weights = []
        for x in reversed(self.training_weights):
            new_list = np.zeros((x.shape[1], x.shape[0]))
            for i in range(x.shape[1]):
                for j in range(x.shape[0]):
                    new_list[i, j] = x[j, i]
            new_weights.append(new_list)
        self.weights = new_weights.copy()
        new_biases = []
        for x in reversed(self.training_biases):
            new_biases.append(x)
        self.biases = new_biases.copy()

    def backpropagation(self, cost_d):
        self.create_training_net()
        new_weights = self.training_weights.copy()
        for x in range(len(new_weights)):
            new_weights[x] = self.d_cost_weight(x, cost_d)
        learning_rate = 0.1
        for x in range(len(new_weights)):
            for i in range(len(self.training_layers[x])):
                for j in range(len(self.training_layers[x + 1])):
                    self.training_weights[x][i, j] -= learning_rate * new_weights[x][i, j]

        new_biases = self.training_biases.copy()
        for x in range(len(new_biases)):
            new_biases[x] = self.d_cost_bias(x, cost_d)
        for x in range(len(new_biases)):
            for i in range(len(new_biases[x])):
                self.training_biases[x][i] -= learning_rate * new_biases[x][i]
        self.update()

    # With this function we set the activation function that we want to use
    # Note: maybe in the future we want to use different activation functions for each layer
    def set_activation_f(self, activation_f):
        self.activation_f = []
        for x in range(len(self.weights)):
            self.activation_f.append(activation_f)


# This contain a training sample then we use a list of this class as training data-set for our neural network
class TrainingData:

    def __init__(self, input_values, output_values):
        self.input = input_values
        self.output = output_values


# some Useful functions
def generate_weights(input_layer, hidden_layer, output_layer):
    weights = []
    if len(hidden_layer) != 0:
        weights.append(np.random.rand(input_layer, hidden_layer[0]))
        if len(hidden_layer) > 1:
            for i in range(len(hidden_layer) - 1):
                weights.append(np.random.rand(hidden_layer[i], hidden_layer[i + 1]))
            else:
                weights.append(np.random.rand(hidden_layer[len(hidden_layer) - 1], output_layer))
        else:
            weights.append(np.random.rand(hidden_layer[0], output_layer))
    else:
        weights.append(np.random.rand(input_layer, output_layer))
    return weights


def generate_biases(weights):
    biases = []
    for x in range(len(weights)):
        biases.append(np.random.sample(weights[x].shape[1]))
    return biases


def next_layer(layer, weights, biases, activation_function):
    return af.activation_f_calculate_array(activation_function, add_biases(np.dot(layer, weights), biases))


def add_biases(layer, biases):
    return [(x + y) for x, y in zip(layer, biases)]


def calculate_err(target, prediction):
    return ((prediction - target) ** 2) / 2


def calculate_err_d(target, prediction):
    return prediction - target
