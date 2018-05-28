# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>
#
# demo_symmetry_detection added by Akira Date
# bugs(?) fixed
# I don't know why this code doesn't work. 

import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

random.seed(0)


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a


def randn(coefficient=0.1):
    return coefficient * random.normalvariate(0.0, 1.0)


# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def tanh(x):
    return math.tanh(x)


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return y * (1.0 - y)


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dtanh(y):
    return 1.0 - y ** 2


class NN:
    def __init__(self, ni, nh, no, title='none'):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh + 1  # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                #self.wi[i][j] = rand(-0.2, 0.2)
                #self.wi[i][j] = rand(-1.0, 1.0)
                self.wi[i][j] = randn()
        for j in range(self.nh):
            for k in range(self.no):
                #self.wo[j][k] = rand(-2.0, 2.0)
                #self.wo[j][k] = rand(-1.0, 1.0)
                self.wo[j][k] = randn()

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
        self.activation = sigmoid
        self.dactivation = dsigmoid
        self.errors = []
        self.iteration_num = None
        self.title = title
        self.epoch = 1

    def update(self, inputs, activation=None):
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        if activation is None:
            activation = self.activation

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh - 1):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = activation(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = activation(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M, dactivation=None):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        if dactivation is None:
            dactivation = self.dactivation

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dactivation(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh - 1):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dactivation(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change
                # print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh - 1):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        self.epoch = len(patterns)
        self.iteration_num = iterations
        for i in tqdm(range(iterations * self.epoch)):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % self.epoch == 0:
                self.errors.append(error)

    def print_error(self, is_graph=False):
        if is_graph is True:
            x = [x for x, _ in enumerate(self.errors)]
            plt.plot(x, self.errors)

            plt.xlabel("epoch")
            plt.ylabel("error")
            plt.title(f'title:{self.title}, '
                      f'perceptron=[{self.ni-1}, {self.nh-1}, {self.no}], \n'
                      f'iter={self.iteration_num}, '
                      f'activation={self.get_activation()}, '
                      f'dactivation={self.get_dactivation()} ')
            plt.ylim(0.0, max(self.errors) + 1.0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(timestamp + '.pdf')
            plt.show()
        else:
            for error in self.errors:
                print('error %-.5f' % error)

    def get_activation(self):
        if self.activation is tanh:
            return "tanh"
        return "sigmoid"

    def get_dactivation(self):
        if self.dactivation is dtanh:
            return 'dtanh'
        return 'dsigmoid'


def demo():
    # Teach network XOR function
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1, title='XOR')
    #n.activation = tanh
    #n.dactivation = dtanh
    # train it with some patterns
    n.train(pat, iterations=10000)
    n.print_error(is_graph=True)
    # test it
    n.test(pat)


def demo_symmetry_detection():
    # Teach network XOR function

    pat = [
        [[0, 0, 0, 0, 0, 0], [1]],
        [[0, 0, 0, 0, 0, 1], [0]],
        [[0, 0, 0, 0, 1, 0], [0]],
        [[0, 0, 0, 0, 1, 1], [0]],
        [[0, 0, 0, 1, 0, 0], [0]],
        [[0, 0, 0, 1, 0, 1], [0]],
        [[0, 0, 0, 1, 1, 0], [0]],
        [[0, 0, 0, 1, 1, 1], [0]],
        [[0, 0, 1, 0, 0, 0], [0]],
        [[0, 0, 1, 0, 0, 1], [0]],
        [[0, 0, 1, 0, 1, 0], [0]],
        [[0, 0, 1, 0, 1, 1], [0]],
        [[0, 0, 1, 1, 0, 0], [1]],
        [[0, 0, 1, 1, 0, 1], [0]],
        [[0, 0, 1, 1, 1, 0], [0]],
        [[0, 0, 1, 1, 1, 1], [0]],
        [[0, 1, 0, 0, 0, 0], [0]],
        [[0, 1, 0, 0, 0, 1], [0]],
        [[0, 1, 0, 0, 1, 0], [1]],
        [[0, 1, 0, 0, 1, 1], [0]],
        [[0, 1, 0, 1, 0, 0], [0]],
        [[0, 1, 0, 1, 0, 1], [0]],
        [[0, 1, 0, 1, 1, 0], [0]],
        [[0, 1, 0, 1, 1, 1], [0]],
        [[0, 1, 1, 0, 0, 0], [0]],
        [[0, 1, 1, 0, 0, 1], [0]],
        [[0, 1, 1, 0, 1, 0], [0]],
        [[0, 1, 1, 0, 1, 1], [0]],
        [[0, 1, 1, 1, 0, 0], [0]],
        [[0, 1, 1, 1, 0, 1], [0]],
        [[0, 1, 1, 1, 1, 0], [1]],
        [[0, 1, 1, 1, 1, 1], [0]],
        [[1, 0, 0, 0, 0, 0], [0]],
        [[1, 0, 0, 0, 0, 1], [1]],
        [[1, 0, 0, 0, 1, 0], [0]],
        [[1, 0, 0, 0, 1, 1], [0]],
        [[1, 0, 0, 1, 0, 0], [0]],
        [[1, 0, 0, 1, 0, 1], [0]],
        [[1, 0, 0, 1, 1, 0], [0]],
        [[1, 0, 0, 1, 1, 1], [0]],
        [[1, 0, 1, 0, 0, 0], [0]],
        [[1, 0, 1, 0, 0, 1], [0]],
        [[1, 0, 1, 0, 1, 0], [0]],
        [[1, 0, 1, 0, 1, 1], [0]],
        [[1, 0, 1, 1, 0, 0], [0]],
        [[1, 0, 1, 1, 0, 1], [1]],
        [[1, 0, 1, 1, 1, 0], [0]],
        [[1, 0, 1, 1, 1, 1], [0]],
        [[1, 1, 0, 0, 0, 0], [0]],
        [[1, 1, 0, 0, 0, 1], [0]],
        [[1, 1, 0, 0, 1, 0], [0]],
        [[1, 1, 0, 0, 1, 1], [1]],
        [[1, 1, 0, 1, 0, 0], [0]],
        [[1, 1, 0, 1, 0, 1], [0]],
        [[1, 1, 0, 1, 1, 0], [0]],
        [[1, 1, 0, 1, 1, 1], [0]],
        [[1, 1, 1, 0, 0, 0], [0]],
        [[1, 1, 1, 0, 0, 1], [0]],
        [[1, 1, 1, 0, 1, 0], [0]],
        [[1, 1, 1, 0, 1, 1], [0]],
        [[1, 1, 1, 1, 0, 0], [0]],
        [[1, 1, 1, 1, 0, 1], [0]],
        [[1, 1, 1, 1, 1, 0], [0]],
        [[1, 1, 1, 1, 1, 1], [1]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(6, 5, 1, title='Mirror')
    # train it with some patterns
    #n.activation = tanh
    #n.dactivation = dtanh
    n.train(pat, iterations=100)
    n.print_error(is_graph=True)
    # test it
    n.test(pat)


class NNSin(NN):
    def __init__(self, ni, nh, no, title='none'):
        super().__init__(ni, nh, no, title)
        self.pattern = None

    def print_error(self, is_graph=False):
        if is_graph is True:
            x = [x for x, _ in enumerate(self.errors)]
            plt.plot(x, self.errors)

            plt.xlabel("epoch")
            plt.ylabel("error")
            plt.title(f'title:{self.title}, '
                      f'perceptron=[{self.ni-1}, {self.nh-1}, {self.no}], \n'
                      f'iter={self.iteration_num}, '
                      f'activation={self.get_activation()}, '
                      f'dactivation={self.get_dactivation()} ')
            plt.ylim(0.0, max(self.errors) + 1.0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(timestamp + '.error.pdf')
            plt.show()

            plt.title(f'title:{self.title}, '
                      f'perceptron=[{self.ni-1}, {self.nh-1}, {self.no}], \n'
                      f'iter={self.iteration_num}, '
                      f'activation={self.get_activation()}, '
                      f'dactivation={self.get_dactivation()} ')
            self.draw()
            for i in [(0, 'red'), (1, 'blue')]:
                x_set = [x[0][0] for x in self.pattern if x[1][0] == i[0]]
                y_set = [y[0][1] for y in self.pattern if y[1][0] == i[0]]
                plt.scatter(x_set, y_set, c=i[1])

            x_set = [x / 10.0 for x in range(-60, 60)]
            y_set = [math.sin(math.pi / 2 * x) for x in x_set]
            plt.plot(x_set, y_set)
            plt.savefig(timestamp + '.draw.pdf')
            plt.show()
        else:
            for error in self.errors:
                print('error %-.5f' % error)

    def draw(self):
        def generate_all_patterns():
            x_set = [x for x in range(-300, 300)]
            y_set = [y for y in range(-75, 75)]
            learn_data = [[x/50, y/50] for x in x_set for y in y_set]
            return learn_data

        patterns = generate_all_patterns()
        result = []
        for p in patterns:
            result.append(self.update(p))
        drow_p = [p for i, p in enumerate(patterns) if result[i][0] <= 0.2]
        draw_x = [x[0] for x in drow_p]
        draw_y = [y[1] for y in drow_p]
        plt.scatter(draw_x, draw_y, marker='.', c='coral', s=0.1)
        drow_p = [p for i, p in enumerate(patterns) if result[i][0] >= 0.8]
        draw_x = [x[0] for x in drow_p]
        draw_y = [y[1] for y in drow_p]
        plt.scatter(draw_x, draw_y, marker='.', c='aqua', s=0.1)


def demo_sin_curve():
    EPOCH = 10000

    def generate_leaning_data(num=100):
        learn_data = []
        for _ in range(num):
            x = rand(-6.0, 6.0)
            y = rand(-1.5, 1.5)
            sin_y = math.sin(math.pi / 2 * x)
            correct = 0
            if y >= sin_y:
                correct = 1
            learn_data.append([[x, y], [correct]])
        return learn_data

    pat = generate_leaning_data()

    n = NNSin(2, 5, 1, title='sin_curve')
    n.pattern = pat
    #n.activation = tanh
    #n.dactivation = dtanh
    n.train(pat, iterations=EPOCH)
    n.print_error(is_graph=True)
    n.test(pat)


if __name__ == '__main__':
    #demo_symmetry_detection()
    #demo()
    demo_sin_curve()