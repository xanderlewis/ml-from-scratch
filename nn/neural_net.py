# A basic (and perhaps naive) implementation of a densely connected neural network ('MLP').
# To be trained using gradient descent with partial derivatives computed via backpropagation.

import numpy as np
import activation_funcs as activation

rng = np.random.default_rng()

class NeuralNet:
	"""A neural network."""
	def __init__(self, layer_sizes):
		# layers is an array of integers representing the size of each layer
		# e.g. [10, 64, 3] would mean: three layers consisting of 10, 64 and 3 neurons respectively
		# The first and last layers are the input and output vectors (non-hidden layers) and so
		# the network itself actually contains n - 1 layer objects.
		# It's better to think of each layer as being some nonlinear transformation between finite-dim vector spaces.
		self.depth = len(layer_sizes)
		self._layers = []

		# Initialise the layers
		for i in range(self.depth - 1):
			self._layers.append(Layer(input_size=layer_sizes[i], output_size=layer_sizes[i+1], f=activation.relu))

	def predict(self, input_vec):
		"""Given a vector of input activations, return the network's current vector of output activations."""
		# The 'forward pass'
		v = input_vec
		# For each layer in the network...
		for layer in self._layers:
			# Feed the input through this layer
			v = layer(v)
		return v

class Layer:
	"""A dense neural network layer."""
	def __init__(self, input_size, output_size, f):
		self.input_size = input_size
		self.output_size = output_size
		# The layer's activation function
		self.f = f

		# The layer's vector of activations
		self.a = np.zeros((output_size,))

		# The layer's trainable parameters: weight matrix and bias vector
		self.W = rng.uniform(-1.0, 1.0, size=(output_size, input_size))
		self.b = np.zeros((output_size))

	def __call__(self, v):
		"""Compute the layer's activations given (usually) the previous layer's activations."""
		return self.f(self.z(v))

	def z(self, v):
		"""Compute *just* the weighted sum (plus bias) of the previous layer's activations."""
		# The layer activation's 'affine part'
		return self.W @ v + self.b