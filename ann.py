"""A CLI for neural network training.

Available training sets:
and, or, xor

Usage:
    ann <training_set> (<layers>... | -w WEIGHTS) [-e EPOCHS] [--sigmoid | --threshold]
    ann (-h | --help)

Options:
    --sigmoid       Use a sigmoid activation function. [default]
    --threshold     Use a threshold activation function.
    -w WEIGHTS      Preload weights for a training set.
    -e EPOCHS       Number of epochs. [default: 50]
    -h --help       Show this screen.
"""

from docopt import docopt

from collections import namedtuple
from namedlist import namedlist
import itertools
import random
import math

import training

def threshold(x):
	return 1 if x >= .5 else 0

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
	return x * (1 - x)

class Node(namedlist('node', ['parent_edges', 'value', 'child_edges'])):
	__slots__ = ()
	def __repr__(self):
		return str(self.value)

class Edge(namedlist('edge', ['parent', 'weight', 'child'])):
	__slots__ = ()
	def __repr__(self):
		return str(self.child)

class Perceptron:
	def __init__(self, learning_rate=.5, layer_sizes=(1, 1), weight_init=lambda: random.random() - .5, activation_fx=sigmoid):
		counter = itertools.count()
		# network = ((nodes), (nodes), (nodes))
		network = tuple(
			tuple(Node([], next(counter), []) for i in range(size)) # layer
			for size in layer_sizes)
		for parents, children in zip(network, network[1:]):
			for parent, child in itertools.product(parents, children):
				edge = Edge(parent, weight_init(), child)
				parent.child_edges.append(edge)
				child.parent_edges.append(edge)

		self.net = network
		self.learning_rate = learning_rate
		self.activation_fx = activation_fx

	def __str__(self):
		return '\n\n'.join(
			('\n'.join(
				'\t'.join(str(edge.weight) for edge in node.child_edges)
				for node in layer
				))
			for layer in self.net[:-1])

	def loads(self, weights):
		for layer, layer_weights in zip(self.net[:-1], weights.strip().split('\n\n')):
			for node, node_weights in zip(layer, layer_weights.strip().split('\n')):
				for edge, edge_weight in zip(node.child_edges, node_weights.strip().split()):
					edge.weight = float(edge_weight)

	def set_inputs(self, ins):
		for node, value in zip(self.net[0], ins):
			node.value = value

	def activate_layers(self):
		for layer in self.net[1:]:
			for node in layer:
				weighted_sum = sum(parent.value * weight for parent, weight, _ in node.parent_edges)
				node.value = self.activation_fx(weighted_sum)

	def train(self, ins, actual_outs):
		expected_outs = self.run(ins)
		errors = tuple(exp - act for exp, act in zip(expected_outs, actual_outs))
		self.back_propagate(errors)
		return actual_outs, expected_outs

	def back_propagate(self, errors):
		for layer, parent_nodes in zip(self.net[:0:-1], self.net[-1::-1]):
			next_errors = (
				sum(edge.weight * error
					for edge in parent.child_edges)
				for parent, error in zip(parent_nodes, errors))
			for node, error in zip(layer, errors):
				for edge in node.parent_edges:
					parent, _, _ = edge
					weight_delta = self.learning_rate * error * d_sigmoid(node.value) * parent.value
					edge.weight += weight_delta
			errors = next_errors

	def run(self, ins):
		self.set_inputs(ins)
		self.activate_layers()
		return tuple(node.value for node in self.net[-1])

def main(args):
	training_set = training.sample(args['<training_set>'])
	epochs = int(args['-e'])
	activation = threshold if args['--threshold'] else sigmoid

	if args['-w'] is not None:
		layers, weights = training.weights(args['-w'])
	else:
		layers, weights = map(int, args['<layers>']), None

	p = Perceptron(layer_sizes=layers, activation_fx=activation)
	if weights is not None:
		p.loads(weights)

	for i in range(epochs):
		for inputs, outputs in training_set():
			p.train(inputs, outputs)

	print('(inputs, outputs, predicted) after {} epochs:'.format(epochs))
	for inputs, outputs in training_set():
		print(inputs, outputs, p.run(inputs))

	print('\nNetwork weights:')
	print(p)

if __name__ == "__main__":
	args = docopt(__doc__)
	main(args)