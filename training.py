"""Contains neural network training sets.
"""

import sys

def sample(data):
	return getattr(sys.modules[__name__], '{}_sample'.format(data))

def weights(data):
	return getattr(sys.modules[__name__], '{}_WEIGHTS'.format(data.upper()))

def xor_sample():	
	yield [0, 0], [0]
	yield [0, 1], [1]
	yield [1, 0], [1]
	yield [1, 1], [0]

XOR_WEIGHTS = (2, 2, 1), """1 -1
		-1 1

		1 1
		"""

def and_sample():
	yield [0, 0], [0]
	yield [0, 1], [0]
	yield [1, 0], [0]
	yield [1, 1], [1]

def or_sample():
	yield [0, 0], [0]
	yield [0, 1], [1]
	yield [1, 0], [1]
	yield [1, 1], [1]
