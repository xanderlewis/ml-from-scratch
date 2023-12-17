# A basic (and perhaps naive) implementation of a neural network.
# Trained using gradient descent with partial derivatives computed via backpropagation.

import numpy as np
rng = np.random.default_rng()

class NeuralNet:
	"""A neural network."""
	def __init__(self, layers):
		self.layers = layers

	def predict(self, input_vec):
		"""Given a vector of input activations, return the network's current vector of output activations."""
		# The 'forward pass'
		v = input_vec
		for layer in self.layers:
			# Compute next layer's activation
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
		"""Compute the layer's activations given the previous layer's activations."""
		return self.f(self.z(v))

	def z(self, v):
		"""Compute *just* the weighted sum (plus bias) of the previous layer's activations."""
		# The layer's 'affine function'
		return self.W @ v + self.b

# Activation functions and their derivatives...

def relu(x):
	return np.max(np.vstack([x, np.zeros_like(x)]), axis=0)

def relu_prime(x):
	# Note: ReLU is technically not differentiable at zero, but we pretend its derivative is zero there anyway.
	return 0 if x <= 0 else 1

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
	return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

# Function you might want to use for a final layer...

def softmax(x):
	es = np.exp(x)
	return es / np.sum(es)