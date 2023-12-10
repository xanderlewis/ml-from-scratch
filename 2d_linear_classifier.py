# A quick and dirty implementation of a linear classifier.

import random
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

# Data (points in the unit square; six of each class)
points = [
	[0.1, 0.2], [0.23, 0.3], [0.2, 0.65], [0.35, 0.45], [0.4, 0.15], [0.48, 0.32],
	[0.4, 0.8], [0.6, 0.7], [0.6, 0.6], [0.7, 0.85], [0.75, 0.75], [0.95, 0.9]]

# Rescale points to lie in [-1, 1] x [-1, 1]
for point in points:
	for i in point:
		i = 2 * i - 1


true_classes = [
	0, 0, 0, 0, 0, 0,
	1, 1, 1, 1, 1, 1]

NUM_DATA_POINTS = len(true_classes)

# Hyperparameters
EPOCHS = 300
LEARNING_RATE = 0.1

# Create our model
model = BinaryLinearClassifier()

# -- Training loop --
# We'll do 'full batch' training. (i.e. 'batch size' is equal to the number of training examples)'
epochs_done = 0

while(True):
	print(f'\n(Epoch {epochs_done})')

	# (I) Inference
	predictions = [model(point[0], point[1]) for point in points]

	print('current predictions:', predictions)

	# (II) Compute loss
	loss = mean_squared_error(predictions, true_classes)
	cprint(f'loss: {loss}', 'red')

	# Compute 'accuracy'
	# Choose the nearest class to each prediction
	predicted_classes = [round_prediction(p) for p in predictions]

	print('predicted classes: ', predicted_classes)
	print('true classes:      ', true_classes)

	# Count the number of correct predictions and calculate a percentage
	percent_accuracy = [i == j for i, j in zip(predicted_classes, true_classes)].count(True) * 100 / NUM_DATA_POINTS
	print(f'accuracy: {percent_accuracy:.1f}%')

	if percent_accuracy == 100.0:
		cprint(f'Converged after {epochs_done} epochs. 👏🏻', attrs=['bold'])
		break

	print('weight 1: ', model.w1)
	print('weight 2: ', model.w2)
	print('bias: ', model.b)

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