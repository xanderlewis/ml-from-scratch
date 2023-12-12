# “Loss functions quantify preferences.”

(over the parameters of the model)

Basically, we’re saying to the model (which can 'learn'): we want *this* number to be small. Try and achieve that however you can (by adjusting your parameters iteratively to get closer to a local (and hopefully global) minimum.

So regularisation is the idea of using the loss function to quantify additional preferences about the model: the outputs should not just be close approximations of the ‘ground truth’ / training outputs; we also prefer certain kinds of models.

It’s not just the predictions that count; it’s also the way we predict.

This is because we don’t really care about learning the training data; we want generalisation.

So we adjust these loss functions so that the model also does well on the validation data (and we also adjust the hyperparameters to make this happen).

Only at the VERY END do we use the testing data to evaluate the model. We **do not** then go and start trying to tweak the hyperparameters, because this would effectively be fitting the model (at a higher level) to the testing data — which is cheating and doesn’t tell you anything useful!

Simpler models generalise better.

It’s kind of like Occam’s razor, but for machine learning.