# The proper OOP manifestation of a basic neural net

Just naively thinking and waffling about this stuff for a moment…

We probably want to have a **NeuralNetwork** class, which will have a number (or an array) of layers, which could themselves usefully be objects. We’d want to expose some method to ‘call’ the model on some input vector to get its output vector, but we’d also like to compare model predictions on some input with its matching ‘target’ output via a loss function. To keep the training and inference parts semantically separate, we ought to have some other class like an **Optimiser**.

An **Optimiser** would be in charge of training a model (in particular a neural net) and so would have a chosen loss function and a chosen gradient descent algorithm. It would make calls to the model to make predicts and use its loss function to compare. The only slight problem with this is that the loss function, as well as for a fixed model being a function of the inputs and target outputs, for *fixed* inputs and target outputs is a function of the model’s parameters: weights and biases. But these parameters lie in the **NeuralNetwork** instance, so it’s not completely obvious to me how to connect these in a clean way. The **Optimiser** should also be able to tweak the parameters of the network according to the gradient it computes… I guess we can just expose an **updateParameters** method or something on the model and pass in either new parameters or changes to these. But should we handle layers separately? Each **Layer** instance would have a **call** function similar to that of the whole model, which just does the usual matrix multiplication, vector addition and (point wise) nonlinear activation function. Ideally we’d like the **Optimiser** to be able to compute the gradients with respect to each layer sequentially by doing backpropagation. Doesn’t seem too hard, but it needs access to not only the output of the network but the activations and (pre-activation) affine function outputs of each layer to do this. So I guess the **Layer** instances should store these for later as well as just computing outputs given inputs.

----
…we could go all the way and have a **Neuron** class (that would each have an activation function, a set of weights, etc.) or something… but if we want to leverage nicely parallelisable (and easier to write down!) vector and matrix computations maybe it’s best not to. For a bog-standard feedforward network this is probably just silly.
----

We want to have decent encapsulation/abstraction so that the network only deals with network-level things (inputs and outputs to the actual whole model), each layer deals with its own parameters and stuff, and the optimiser deals with the (as far as is possible) abstract task of optimising the network parameters to minimise the loss function. It shouldn't 'know' anything about the actual internal details of the model, other than that is required to compute loss and its gradient.

So we'll have something like this...

```
class NeuralNetwork:
	def __init__(self):
		self.layers = []

	def __call__(self):
		"""Compute the network's output on a given vector of input activations."""
		# It'll call on its layers in sequence to do the appropriate computations and spit out the result.
		...

class Layer:
	def __init__(self, input_size, output_size, activation):

		self.z = np.empty((output_size,)) # the affine part of the function defining layer activation; needed for backprop
		self.a = np.empty((output_size,)) # the activation vector of this layer (so, activation(z))
		self.W = np.zeros(output_size, input_size)) # (or however we want to initialise our weight matrix)
		self.b = np.zeros((output_size,))
		self.activation = activation # I guess usually will be ReLU

	def __call__(self, x):
		"""Compute the layer's output on a given input (usually from the previous layer)."""
		return self.activation(matmul(x, self.W) + b) # or whatever
		# but we want to store the intermediate result self.z
		# so maybe do:
		self.z = matmul(x, self.W) + b # affine part
		self.a = self.activation(self.z)
		return self.a
		...

	def update_parameters(self, delta_W, delta_b):
		"""Pass in some updates to the weights and bias vector; updates the layer by adding them pointwise."""
		# e.g. just pass in -1 * grad (wrt params) * LEARNING_RATE or whatever


class Optimiser:
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

```

How should we deal with the mini-batching of training data? We can feed in a huge lump of training examples (x_train, y_train) pairs -- huge matrices, but if we want to do *stochastic* gradient descent, we better split this data up into batches and repeat out gradient computations on these individually. I guess we could do something like I did before... having a **BatchManager** class or something that holds some training data and shuffles it and spits out batches when asked. That seems a nice way to keep it all neat.