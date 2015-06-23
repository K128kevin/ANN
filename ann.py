"""A CLI for neural network training.

Usage:
    ann

Options:
    -h --help        Show this screen.
    --version        Show version.
"""

from docopt import docopt

from collections import namedtuple
from namedlist import namedlist
import itertools
import random
import math

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
	def __init__(self, learning_rate=.5, layer_sizes=(1, 1), weight_init=lambda: random.random() - .5):
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

	def __str__(self):
		return '\n\n'.join(
			('\n'.join(
				'\t'.join(str(edge.weight) for edge in node.child_edges)
				for node in layer
				))
			for layer in self.net[:-1])

	def set_inputs(self, ins):
		for node, value in zip(self.net[0], ins):
			node.value = value

	def activate_layers(self):
		for layer in self.net[1:]:
			for node in layer:
				weighted_sum = sum(parent.value * weight for parent, weight, _ in node.parent_edges)
				node.value = sigmoid(weighted_sum)

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
	p = Perceptron(layer_sizes=(2, 2))
	for i in range(500):
		a = p.train([1, 1], [0, 0])
		b = p.train([0, 1], [1, 1])
		a = p.train([1, 0], [0, 0])
		b = p.train([0, 0], [1, 1])
		if i % 50 == 0:
			print(a)
			print(b)
			print(p)

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)