"""A CLI for neural network training.

Available training sets:
and, or, xor

Example:
	ann or 2 1 --threshold -e 100

Usage:
    ann <training_set> (<layers>... | -w WEIGHTS) [-l L_RATE] [-e EPOCHS] [--sigmoid | --threshold]
    ann (-h | --help)

Options:
    --sigmoid       Use a sigmoid activation function. [default]
    --threshold     Use a threshold activation function.
    -w WEIGHTS      Preload weights for a training set.
    -l L_RATE       Learning rate. [default: .05]
    -e EPOCHS       Number of epochs. [default: 50]
    -h --help       Show this screen.
"""

from docopt import docopt

from collections import namedtuple
import itertools
import random
import math
from io import StringIO
import numpy as np

import training

def threshold(x):
	return 1 if x > .5 else 0

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

class FeedForwardNet:
	def __init__(self, layer_sizes=(1, 1), activation_function=sigmoid):
		self.input_size = layer_sizes[0]
		self.weights = [np.zeros([b, a]) for a, b in zip(layer_sizes[:-1], layer_sizes[1:])]
		self.act_fn = np.vectorize(activation_function)
		self.d_act = np.vectorize(d_sigmoid)

	def loads(self, weights):
		self.weights = [np.loadtxt(StringIO(m), ndmin=2) for m in weights.split('\n\n')]
		self.input_size = self.weights[0].shape[1]

	def run(self, inputs):
		if len(inputs) != self.input_size:
			raise Exception('Wrong input size')
		inputs = np.array(inputs).reshape(len(inputs), 1)
		computation = [inputs] + self.weights
		self.layers = list(itertools.accumulate(computation, lambda A, B: self.act_fn(np.dot(B, A))))
		return self.layers[-1].copy()

	def train(self, inputs, outputs, l_rate=.05):
		expected = self.run(inputs)
		errs = expected - outputs
		for ins, outs, weights in zip(self.layers[-2::-1], self.layers[:0:-1], reversed(self.weights)):
			weight_delta = -l_rate * self.d_act(outs) * errs.dot(ins.reshape(1, len(ins)))
			weights += weight_delta
			errs = sum(weight_delta) / len(outs)
			errs.resize(len(errs), 1)

def main(args):
	training_set = training.sample(args['<training_set>'])
	l_rate = float(args['-l'])
	epochs = int(args['-e'])
	activation = threshold if args['--threshold'] else sigmoid

	if args['-w'] is not None:
		layer_sizes, weights = training.weights(args['-w'])
	else:
		layer_sizes, weights = list(map(int, args['<layers>'])), None

	p = FeedForwardNet(layer_sizes=layer_sizes, activation_function=activation)
	if weights is not None:
		p.loads(weights)


	print('Initial network weights:')
	print(p.weights)
	print()

	for i in range(epochs):
		for inputs, outputs in training_set():
			p.train(inputs, outputs, l_rate=l_rate)
	print('(inputs, outputs, predicted) after {} epochs:'.format(epochs))
	for inputs, outputs in training_set():
		print(inputs, outputs, p.run(inputs))

	print()
	print('Final network weights:')
	print(p.weights)

if __name__ == "__main__":
	args = docopt(__doc__)
	main(args)