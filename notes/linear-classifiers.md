Another way of thinking of a linear classifier, at least the way I've implemented it, is just as a densely (totally) connected neural network with *no* hidden layers and a linear activation function. Each component of the output activations vector is simply a weighted (& biased) sum of the inputs. The 2D one just looks something like:
 ________
|_input x|---w_x---\                _____________________________
 ________           --- (+ b) ---->|__output (class prediction)__|
|_input y|---w_y---/

The training algorithm -- (stochastic) gradient descent -- is exactly as it is for any other neural net. It happens that computing the derivative of the loss function with respect to these three parameters is more straightforward than in most cases, but it's the same thing.