# All of the below are vectorised.

# -- Activation functions and their derivatives --

def relu(x):
	"""Computes componentwise reLU of a vector."""
	return np.max(np.vstack([x, np.zeros_like(x)]), axis=0)

def relu_prime(x):
	# Note: ReLU is technically not differentiable at zero, but we pretend its derivative is zero there anyway.
	return 0 if x <= 0 else 1

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
	return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

# -- Function you might want to use for a final layer --

def softmax(x):
	es = np.exp(x)
	return es / np.sum(es)