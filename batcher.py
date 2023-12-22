import numpy as np

class BatchManager:
	"""Shuffles a given dataset and returns mini batches of the desired length."""
	def __init__(self, xs, ys):
		assert xs.shape[0] == ys.shape[0]
		self.xs = xs
		self.ys = ys
		# (the above should be 2D ndarrays with matching axis 0 length)
		# (i.e. a pair of matrices with the same number of rows)
		# xs.shape[1] is the input vector dimension; ys.shape[1] is the output vector dimension

		self.reset()

	def reset(self):
		# Shuffle the data ready for splitting into 'mini' batches
		p = np.random.permutation(self.xs.shape[0])
		self.xs = self.xs[p, :] # (using 'advanced indexing')
		self.ys = self.ys[p, :]

		# Start from the first row of xs and ys
		self._batch_pointer = 0

	def next_mini_batch(self, batch_size):
		if self._batch_pointer >= self.xs.shape[0]:
			# No more mini batches
			return None
		else:
			from_row = self._batch_pointer
			to_row = self._batch_pointer + batch_size
			self._batch_pointer += batch_size

			# We return rows from_row up to (and not including) to_row
			# Fails gracefully when there aren't quite that many rows left
			return (self.xs[from_row : to_row, :], self.ys[from_row : to_row, :])