from linear_classifier import LinearClassifier
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Get linear classifier I trained earlier on MNIST
with open('mnist_linear_model.pickle', 'rb') as f:
	model = pickle.load(f)

# Visualise its 28^2 x 10 weight matrix as a collection of ten 28 x 28 images
for i in range(10):
	plt.subplot(2, 5, i + 1)
	weights = (model.W + model.b)[ : , i].reshape(28, 28)
	plt.title(i)
	plt.imshow((weights), cmap='RdPu')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_visible(False)
	frame1.axes.get_yaxis().set_visible(False)
plt.show()