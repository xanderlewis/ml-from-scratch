import numpy as np

class KNearestClassifier:
	"""A k-nearest neighbours classifier. Takes n-dimensional inputs and outputs an m-dimensional vector of class probabilities."""
	# Basically, K-nearest neighbours memorises the training data.
	# Given an input, it then looks for the closest k inputs it saw during training, and chooses a prediction of the class
	# based on the (ground truth) classes these inputs had. It could do this by a majority vote, or taking an average of
	# scores, or something else. If we have a tie, maybe choose the closest or just choose randomly.
	# For straightforward **nearest neighbour classification**, we just pick the label of the closest sample in the training set.

	def __init__(self, k, mode='majority'):
		self.k = k

		# (mode can be 'majority' or 'average')

	def train(self):
		"""'Train' the classifier on a set of inputs (samples) and their correct outputs (labels)."""
		pass

	def __call__(self, input):
		"""Use the model to predict the labels of a given set of inputs."""
		pass

if __name__ == '__main__':
	pass