# A quick and dirty implementation of a linear classifier.

import random
import numpy as np # (for random number generation)
import matplotlib.pyplot as plt
from termcolor import cprint

class BinaryLinearClassifier:
	"""A linear classifier. Input vectors are two-dimensional; output is a scalar."""

	def __init__(self, seed=None):
		# Initialise learnable parameters
		random.seed(seed)
		self.w1 = random.uniform(-1.0, 1.0)
		self.w2 = random.uniform(-1.0, 1.0)
		self.b = 0.0

	def __call__(self, x1, x2):
		"""Call the model (perform an inference) on a given input vector."""
		# The output is a weighted sum of the input components, plus a scalar bias.
		return self.w1*x1 + self.w2*x2 + self.b

	# [The below methods assume we're using a loss function of the form (y_p - y) ** 2]
	def grad_w1(self, x1, x2, y):
		return 2 * (self.w1*x1 + self.w2*x2 + self.b - y) * x1

	def grad_w2(self, x1, x2, y):
		return 2 * (self.w1*x1 + self.w2*x2 + self.b - y) * x2

	def grad_b(self, x1, x2, y):
		return 2 * (self.w1*x1 + self.w2*x2 + self.b - y)


def mean(l):
	return sum(l) / len(l)

def mean_squared_error(ys_pred, ys):
	"""Returns the mean squared error of a set of predicted classes, given the true classes."""
	return mean([(i - j) ** 2 for i, j in zip(ys_pred, ys)])

def round_prediction(p):
	if p >= 0.5:
		return 1
	else:
		return 0

# (Ideally we want to have two sets of points that are actually linearly separable)
def generate_points(n, std_dev=0.2):
	"""Returns a 2D 'point cloud' consisting of two randomly-generated clusters of n points in the given square."""
	# Centered at (loc1, loc1) and (loc2, loc2)
	loc1 = -0.5
	loc2 = 0.5
	class1 = []
	class2 = []

	for i in range(n):
		class1.append(list(rng.normal(loc1, std_dev, 2)))
		class2.append(list(rng.normal(loc2, std_dev, 2)))

	return class1 + class2

def sample_model_over_grid(density=16):
	"""Samples values from the model at each point in a square grid and returns a numpy array."""
	surface = [[model(i, j) for j in np.linspace(-1, 1, density)] for i in np.linspace(1, -1, density)]

	return surface

def plot_prediction_surface():
	surface = sample_model_over_grid(density=32)
	s = STANDARD_DEVIATION * 2
	plt.imshow(surface, cmap='plasma', extent=[-1 - s,1 + s,-1 - s,1 + s])

def plot_data_points():
	plt.scatter([p[0] for p in points], [p[1] for p in points], c='white', s=1)

rng = np.random.default_rng()

# Data parameters
NUM_EACH_CLASS = 100
STANDARD_DEVIATION = 0.2

# Model hyperparameters
EPOCHS = 300
LEARNING_RATE = 0.15

# Generate training (and, in this case, testing) data
points = generate_points(n=NUM_EACH_CLASS, std_dev=STANDARD_DEVIATION)
true_classes = [0] * NUM_EACH_CLASS + [1] * NUM_EACH_CLASS

# LATER: plot the initial (randomly initialised) and final decision boundaries.
# (plot w1 * x1 + w2 * x2 + b = 0.5)

# Create our model
model = BinaryLinearClassifier()

# Save the initial model parameters
initial_w1 = model.w1
initial_w2 = model.w2
initial_b = model.b

plt.ion()

# -- Training loop --
# We'll do 'full batch' training. (i.e. 'batch size' is equal to the number of training examples)
epochs_done = 0

while(True):
	print(f'\n(Epoch {epochs_done})')

	# (I) Inference
	predictions = [model(point[0], point[1]) for point in points]

	#print('current predictions:', predictions)

	# (II) Compute loss
	loss = mean_squared_error(predictions, true_classes)
	cprint(f'loss: {loss}', 'red')

	# TODO: add regularisation penalty to loss function (then I need to change the gradient formulae accordingly)

	# Compute 'accuracy'
	# Choose the nearest class to each prediction
	predicted_classes = [round_prediction(p) for p in predictions]

	#print('predicted classes: ', predicted_classes)
	#print('true classes:      ', true_classes)

	# Count the number of correct predictions and calculate a percentage
	percent_accuracy = [i == j for i, j in zip(predicted_classes, true_classes)].count(True) * 100 / (NUM_EACH_CLASS * 2)
	cprint(f'accuracy: {percent_accuracy:.2f}%', 'blue')

	if percent_accuracy == 100.0:
		cprint(f'Converged after {epochs_done} epochs. 👏🏻', attrs=['bold'])
		cprint(f'\nInitial model parameters: \nw1 = {initial_w1} \nw2 = {initial_w2} \nb = {initial_b}', 'yellow')
		cprint(f'\nFinal model parameters: \nw1 = {model.w1} \nw2 = {model.w2} \nb = {model.b}', 'magenta')
		input('Press enter to quit.')
		quit()

	#print('weight 1: ', model.w1)
	#print('weight 2: ', model.w2)
	#print('bias: ', model.b)

	# Plot stuff
	plt.clf()
	plot_prediction_surface()
	plot_data_points()
	plt.show()
	plt.pause(0.01)

	# (III) Compute gradient of loss wrt model parameters
	# (Differentiation is linear ==> derivative of the mean is the mean of the derivatives)
	g_w1 = mean([model.grad_w1(x[0], x[1], y) for x, y in zip(points, true_classes)])
	g_w2 = mean([model.grad_w2(x[0], x[1], y) for x, y in zip(points, true_classes)])
	g_b = mean([model.grad_b(x[0], x[1], y) for x, y in zip(points, true_classes)])

	# (IV) Update parameters
	model.w1 -= LEARNING_RATE * g_w1
	model.w2 -= LEARNING_RATE * g_w2
	model.b -= LEARNING_RATE * g_b

	epochs_done += 1

	#input()