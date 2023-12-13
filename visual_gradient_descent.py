# A little tool for visualising gradient descent on a scalar function of two variables.
# The idea is that we can create multiple 'descenders' with different rules and see how they fare on different functions.
# For now it's all 2D; might add 3D graphing later.
# Also, some way to actually save settings, edit them and rerun would be nice.

import tensorflow as tf
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from termcolor import cprint
from enum import Enum

# Grid for plotting on the unit square
RES = 512
x = np.linspace(0, 1, RES)
y = np.linspace(0, 1, RES)
X, Y = np.meshgrid(x, y)


class Rule(Enum):
	VANILLA = 0
	SGD = 1
	SGD_MOMENTUM = 2

class ScalarField:
	def __init__(self, f, cmap):
		self.f = f
		self.cmap = cmap

	def show(self):
		plt.pcolormesh(X, Y, self.f(X, Y), cmap=self.cmap)


class Descender:
	"""A thing that tries to find a minimum by using gradient descent."""
	def __init__(self, starting_x, starting_y, field, rule, eta=0.005, colour='0.2'):
		self.x = starting_x
		self.y = starting_y
		self.rule = rule
		self.history = [(starting_x, starting_y)]
		self.field = field
		self.eta = eta
		self.colour = colour

	def update(self):
		"""Update the position according to the 'rule' and the given function."""
		# Compute the new position by evaluating the gradient at the current point, and add this to history
		
		# Make tensor copies of x and y coords
		xt = tf.Variable(initial_value=self.x)
		yt = tf.Variable(initial_value=self.y)

		# Calculuate gradient vector using tf.GradientTape
		with tf.GradientTape(persistent=True) as tape:
			value = self.field.f(xt, yt)
		grad_x = float(tape.gradient(value, xt))
		grad_y = float(tape.gradient(value, yt))

		# If simulating SGD, introduce some noise
		if self.rule == Rule.SGD:
			grad_x += random.uniform(-1, 1) * 2
			grad_y += random.uniform(-1, 1) * 2

		# Update position according to gradient
		self.history.append((self.x, self.y))
		self.x -= self.eta * grad_x
		self.y -= self.eta * grad_y

	def show(self):
		# Show the full history of the descender
		xs =[point[0] for point in self.history]
		ys =[point[1] for point in self.history]

		plt.plot(xs, ys, color=self.colour, linewidth=0.8, marker='o', markersize=1)

def parse_function(s):
	s = s.replace('^', '**')
	s = s.replace('sin', 'tf.sin')
	s = s.replace('cos', 'tf.cos')
	s = s.replace('tan', 'tf.tan')
	return eval('lambda x, y: ' + s)

def get_user_input():
	# Get function from user
	input_func = parse_function(input('\nWhat function would you like to minimise? [scalar-valued function of (x,y)]\n'))
	iterations = int(input('How many iterations? '))

	# Create field
	field = ScalarField(input_func, 'Spectral')

	# Get info about descenders
	n = int(input('How many descenders? '))

	# Create descenders
	descenders = []
	for i in range(n):
		cprint(f'\n[configuring descender {i + 1}]', attrs=['bold'])
		sx = float(input('Starting x: '))
		sy = float(input('Starting y: '))
		r = input('Rule? [v (vanilla), sgd] ')
		if r == 'v':
			r = Rule.VANILLA
		elif r == 'sgd' or r == 's':
			r = Rule.SGD

		descenders.append(Descender(sx, sy, field, r))

	return field, descenders, iterations

def HARDCODED_TEST_STUFF():
	# Create a scalar field to attempt to minimise the value of
	field = ScalarField(lambda x, y: tf.sin(6 * x) * tf.cos(6 * y), 'Spectral')
	# tf.sin(6 * x) * tf.cos(6 * y)
	# x ** 2 + 5 * y ** 3
	# -x * (-y) ** 3 - (-y) * (-x) ** 3
	# x ** 2 + y ** 2

	# Create some descenders
	descenders = []
	descenders.append(Descender(0.15, 0.70, field, Rule.SGD))
	descenders.append(Descender(0.25, 0.20, field, Rule.SGD, colour='slategrey'))
	descenders.append(Descender(0.35, 0.90, field, Rule.SGD, colour='navy'))
	descenders.append(Descender(0.70, 0.35, field, Rule.SGD, colour='0.3'))

	return field, descenders, 50

if __name__ == '__main__':
	while True:
		# Get settings from user
		field, descenders, iterations = get_user_input()

		# Set up plotting stuff
		plt.ion()
		plt.figure(figsize=(7,7))
		plt.autoscale(False)

		# Draw loop
		for i in range(iterations):
			# Show stuff
			plt.cla()
			field.show()
			for each in descenders:
				each.show()

			# Plot stuff
			ax = plt.gca()
			ax.set_xlim([0.0, 1.0])
			ax.set_ylim([0.0, 1.0])
			plt.show()
			plt.pause(0.2)

			# Update stuff
			for each in descenders:
				each.update()
		plt.ioff()
		plt.show()