"""Contains neural network training sets.
"""

import sys

def dispatch(sample):
	return getattr(sys.modules[__name__], '{}_sample'.format(sample))

def xor_sample():	
	yield [0, 0], [0]
	yield [0, 1], [1]
	yield [1, 0], [1]
	yield [1, 1], [0]

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
