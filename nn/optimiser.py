# An optimiser that can compute derivative information about a neural net and use this to
# perform gradient descent to minimise its loss function.

import numpy as np
import batch_manager as bm

class NeuralNetOptimiser:
	def __init__(self, loss, algo):
		self.loss = loss # choice of loss function e.g. mean squared error; takes a pair of vectors and returns a scalar
		self.algo = algo # choice of optimisation algorithm (e.g. SGD)

	def train(self, model, train_x, train_y epochs, batch_size):
		"""Train the given model on a set of input and (target) output vectors."""
		# -- Implement the basic training loop in here. --
		# Create a batch manager and give it the data and the batch size.
		# Repeatedly:
		# Take out a batch of inputs and outputs, run the inputs through the model (who calls on its layers to do stuff).
		# Use the loss function to compare to the outputs, and do backprop to get the gradients wrt stuff.
		# Call on the model/layers(?) to update their parameters appropriately.

def mean_squared_error(y_preds, ys):
	return np.mean(np.square(y_preds - ys))