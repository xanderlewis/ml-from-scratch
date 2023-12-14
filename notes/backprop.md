# Backpropagation

[very unorganised quick note]

In order to iteratively apply gradient descent to (hopefully) minimise the loss function given a particular training example, we need to *compute* the gradient. That is, we need to find out the partial derivatives of the loss with respect to each of the model's learnable parameters (the weights and biases, in the case of a neural net). To do this, we start by considering partial derivatives of the loss with respect to the outputs of the network, and then derivatives of the outputs with respect to the input to their neuron's activation functions, then of these inputs to the activation functions (the weighted sums of activations in the previous layer) with respect to their inputs (the activations, the biases and their weights)...

...and so on, applying the chain rule to multiply all the derivatives along paths in the network together to hopefully end up with the derivative of loss with respect to each parameter.

That's the mathematics of it, anyway.

The idea of the actual backpropagation 'algorithm' is to recursively compute 'error vectors' for each layer. Given an 'error' for each neuron in layer n + 1, we can compute the 'errors' for the neurons in layer n by looking at the weights connecting the layers. Thinking about it this way, we end up with simple formulae involving matrix multiplications and transposes and stuff that we can easily implement in code.

The error vector for a layer is just the vector whose nth component is the partial derivative of loss with respect to the input (weighted sum) to the nth neuron of that layer. So it's basically the vector of derivatives wrt that layer's outputs, pre-activation.

the vector of partial derivatives of loss with respect to the weighted sums? so each of the inputs to that layer's neurons (before they get put through the activation function for that layer)?

so then to get derivatives with respect to some weight or bias, we just multiply this error by the derivative of the sum with respect to that parameter. In the case of a bias, it's just 1 (derivative of (wa + b) wrt b is 1!). In the case of a weight, it's just the corresponding activation in the previous layer (derivative of (wa + b) wrt w is a!).

So, actually, because of the above facts about derivatives, the components of the error vectors *are* the partial derivatives of loss with respect to the biases!

Similarly, the matrix of partial derivatives with respect to the weights of a layer (so, the gradient of loss viewed as a scalar function of those weights) is just a simple matrix product involving the error vector and the (transpose of the) vector of activations of the previous layer.

As an aside, something I hadn't really thought about before:

Matrix multiplication of the form [row vector] x [column vector] (1xn and nx1) is just like a dot product of the two vectors; in fact if u and v are column vectors then u^{T} x v is precisely u・v.

What about matrix multiplication of the form [column vector] x [row vector] (mx1 and 1xn)? Note that we have more freedom here: the vectors need not have the same 'length'. Well, the result is just a vector of all the possible pairwise products. Kind of a useful thing. We can take the transpose as before to get this for two arbitrary column (/row) vectors.

[Wikipedia](https://en.wikipedia.org/wiki/Backpropagation) does a really good job of explaining all this stuff.