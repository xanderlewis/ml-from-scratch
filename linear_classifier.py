import numpy as np
import matplotlib.pyplot as plt
import pickle
from termcolor import cprint

class LinearClassifier:
	"""A linear classifier mapping between two ('feature space' and 'output space') finite-dimensional Euclidean spaces."""
	def __init__(self, input_dim, output_dim, seed=None):
		# Initialise model parameters
		#self.W = np.random.uniform(-1.0, 1.0, size=(input_dim, output_dim))
		self.W = np.zeros((input_dim, output_dim))
		self.b = np.zeros((1, output_dim))

	def __call__(self, inputs):
		"""Perform inference on an (n by m) matrix consisting of n rows of m-dimensional input vectors."""
		# Matrix multiplication of inputs with W, plus the bias vector
		# (hopefully the bias vector will be broadcast (so added to each row)
		# In the above, 'n' can be anything. inputs could be a row vector representing a single input.
		# That is, we can call it on arbitrary batch sizes.

		# HACKY FIX!!!:
		thing_to_return = (inputs @ self.W) + self.b
		if thing_to_return.shape == (10,):
			thing_to_return = thing_to_return.reshape(1, 10)

		return thing_to_return

		# ** A row in the input matrix (a sample) is dotted with a column in the weight matrix (a weighting)
		# ** to get an entry of the output matrix corresponding to that sample's score for the chosen class

def batch_mean_squared_error(y_preds, ys):
	"""Computes the mean squared errors of each prediction in a batch."""
	# Input is a shape (BATCH_SIZE, 10) array; output is a shape (BATCH_SIZE,) vector
	return np.mean((y_preds - ys) ** 2, axis=1)

def batch_loss(y_preds, ys):
	"""Computes the average loss across across a whole batch."""
	# (Possibly introduce a regularisation penalty term)
	return np.mean(batch_mean_squared_error(y_preds, ys), axis=0)

class BatchManager:
	"""Takes a set of training examples and labels, and produces batches of the specified size."""
	def __init__(self, x_train, y_train, batch_size):
		self.xs = x_train.copy()
		self.ys = y_train.copy()
		self.batch_size = batch_size
		self.num_samples = self.xs.shape[0]
		self.num_batches = np.ceil(self.num_samples / self.batch_size)
		self.reset()
		

	def reset(self):
		self.current_batch = 0

		# Shuffle the (rows of the) data (SIMULTANEOUSLY)
		perm = np.random.permutation(self.xs.shape[0])
		self.xs = self.xs[perm]
		self.ys = self.ys[perm]

		# THE BELOW IS VERY, VERY WRONG!
		#np.random.shuffle(self.xs)
		#np.random.shuffle(self.ys)

	def next_batch(self):
		#cprint(f'Getting batch {self.current_batch}...', 'blue')
		c = self.current_batch
		s = self.batch_size
		new_batch = (self.xs[c * s : (c + 1) * s, :], self.ys[c * s : (c + 1) * s, :])

		self.current_batch += 1

		return new_batch

def one_hot_encode(y):
	"""Returns a 10-dim one-hot encoding of a given output value in {0, ..., 9}."""
	return [0] * y + [1] + [0] * (10 - y - 1)

def prepare_mnist():
	# Load MNIST data [shapes (60000, 28, 28) and (60000,)]
	from keras.datasets import mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Reshape train and test feature vectors (we want each sample to be a 60000-dim vector / rank-1 tensor)
	x_train = x_train.reshape(60000, 28 * 28)
	x_test = x_test.reshape(10000, 28 * 28)

	# Normalise feature vectors (0~255 int |--> 0.0~0.1 float)
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# Create one-hot encodings of train and test output vectors
	y_train = np.array([one_hot_encode(y) for y in y_train])
	y_test = np.array([one_hot_encode(y) for y in y_test])

	return (x_train, y_train), (x_test, y_test)

def round_single_pred(pred):
		if pred >= 0.5:
			return 1
		else:
			return 0

def round_y_preds(ys):
	# return an array where each row is the one-hot encoding of the original row's arg max
	arg_maxes = np.argmax(ys, axis=1)
	return np.array([one_hot_encode(n) for n in arg_maxes]).reshape(60000, 10)

	# BELOW: previously I was just rounding each class prediction towards 0 or 1; whichever was nearest.
	#return np.vectorize(round_single_pred)(ys)

def loss_grad_single_wrt_W(x, y, model):
	"""Computes the gradient of the loss with respect to W on a single training example."""
	# Returns a W-shaped matrix of partial derivatives
	# W should have shape (28 * 28, 10), so l should range through 0 to 28*28 -1, and m through 0 to 9
	
	return np.fromfunction(lambda l, m: (model(x)[0, m.astype(int)] - y[m.astype(int)]) * x[l.astype(int)] / 5, shape=model.W.shape)


def loss_grad_single_wrt_b(x, y, model):
	"""Computes the gradient of the loss with respect to b on a single training example."""
	# Returns a b-shaped vector of partial derivatives
	# (I could probably np.vectorize the gradient function instead)
	return np.fromfunction(lambda l, m: (model(x)[0, m.astype(int)] - y[m.astype(int)]) / 5, shape=model.b.shape)

def loss_grad_wrt_W(x_batch, y_batch, model):
	"""Computes the gradient of the loss function with respect to the weight matrix W on a batch of samples."""
	# Returns an array with same shape as W.
	# Take a batch of xs and ys, compute the gradient of loss wrt W on each row and return the componentwise average
	total = np.zeros((model.W.shape))
	num_samples = x_batch.shape[0]
	for row_index in range(num_samples):
		total += loss_grad_single_wrt_W(x_batch[row_index, :], y_batch[row_index, :], model)
	return total / num_samples


def loss_grad_wrt_b(x_batch, y_batch, model):
	"""Computes the gradient of the loss function with respect to the bias vector b."""
	# Returns an array with the same shape as b.
	total = np.zeros((model.b.shape))
	num_samples = x_batch.shape[0]
	for row_index in range(num_samples):
		total += loss_grad_single_wrt_b(x_batch[row_index, :], y_batch[row_index, :], model)
	return total / num_samples

def loss_grad_check_W_component(x_batch, y_batch, model, component, delta=1e-7):
	"""Computes an approximation to the gradient of loss wrt a single component of W"""
	# Takes a batch of inputs and target outputs, feeds the inputs into the model with translated versions of W
	# and compares.
	old_loss = batch_loss(model(x_batch), y_batch)

	delta_matrix = np.zeros(shape=model.W.shape)
	delta_matrix[component[0], component[1]] = delta

	model.W += delta_matrix
	new_loss = batch_loss(model(x_batch), y_batch)
	model.W -= delta_matrix

	return (new_loss - old_loss) / delta

def loss_grad_check_b_component(x_batch, y_batch, model, component, delta=1e-7):
	"""Computes an approximation to the gradient of loss wrt a single component of b"""
	old_loss = batch_loss(model(x_batch), y_batch)

	delta_vector = np.zeros(shape=model.b.shape)
	delta_vector[component] = delta

	model.b += delta_vector
	new_loss = batch_loss(model(x_batch), y_batch)
	model.b -= delta_vector

	return (new_loss - old_loss) / delta


# Demo the classifier by training it on MNIST
if __name__ == '__main__':
	# Model hyperparameters
	EPOCHS = 10
	BATCH_SIZE = 128
	LEARNING_RATE = 0.005

	# Load MNIST dataset
	(x_train, y_train), (x_test, y_test) = prepare_mnist()

	# x_train has shape (60000, 28 * 28); y_train has shape (60000, 10)
	# x_test has shape (10000, 28 * 28); y_test has shape (10000, 10)

	# Create model and a batch manager
	model = LinearClassifier(28 * 28, 10)
	batch_manager = BatchManager(x_train, y_train, BATCH_SIZE)

	# -- Training loop --
	for epoch in range(EPOCHS):
		print(f'(epoch {epoch})')

		# Compute loss on whole training set
		all_y_preds = model(x_train)
		loss = batch_loss(all_y_preds, y_train)
		cprint(f'loss: {loss}', 'red')

		# Compute accuracy on whole training set
		rounded_y_preds = round_y_preds(all_y_preds)
		num_correct = [(rounded_y_preds[i, :] == y_train[i, :]).all() for i in range(rounded_y_preds.shape[0])].count(True)
		accuracy = (num_correct / rounded_y_preds.shape[0]) * 100
		cprint(f'current accuracy: {accuracy:.2f}%', 'green')

		batch_counter = 0

		# -- START EPOCH --
		while True:
			# (I) ATTEMPT TO GET NEXT (MINI) BATCH OF TRAINING DATA
			x_batch, y_batch = batch_manager.next_batch()

			# Finish the epoch if there are no training samples left
			if x_batch.shape[0] == 0:
				print('batch empty.')
				break

			# (II) RUN THE MODEL
			y_preds = model(x_batch) # (Feed in BATCH_SIZE samples, ask for BATCH_SIZE predictions each of length 10)

			# (III) COMPUTE LOSS (of batch)
			if batch_counter % 100 == 0:
				b_loss = batch_loss(y_preds, y_batch)
				cprint(f'batch loss: {b_loss}', 'red')

			# (IV) COMPUTE GRADIENT FOR CURRENT BATCH
			grad_wrt_W = loss_grad_wrt_W(x_batch, y_batch, model)
			grad_wrt_b = loss_grad_wrt_b(x_batch, y_batch, model)

			# (GRAD CHECK STUFF (ON A SINGLE COMPONENT OF W))
			#check = loss_grad_check_W_component(x_batch, y_batch, model, (400, 3))
			#cprint(f'grad wrt W component check:\t{check}', 'magenta')
			#print(f'real grad wrt W component:\t{grad_wrt_W[400, 3]}')
			#cprint(f'approximation error: {check - grad_wrt_W[400, 3]}', 'red')

			# (V) ADJUST MODEL PARAMETERS ACCORDINGLY
			model.W -= LEARNING_RATE * grad_wrt_W
			model.b -= LEARNING_RATE * grad_wrt_b

			batch_counter += 1

		# No more batches; finish the epoch.
		batch_manager.reset()
		batch_counter = 0

	# Pickle the resulting trained model
	with open('mnist_linear_model.pickle', 'wb') as f:
		pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)



