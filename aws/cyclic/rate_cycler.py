from keras import backend as K
from keras.callbacks import Callback
import numpy as np

class CyclicLR(Callback):
	""" This callback implements a cyclical learning rate policy. It cycles the
	learning rate between two boundaries, as detailed in this paper
	(https://arxiv.org/abs/1506.01186).
	This simplified implementation uses only a stepped triangular policy.
	"""
	def __init__(self, epochs, num_samples, batch_size, scale_factor=0.5, base_lr=10**(-2.5), max_lr=1e-1):
		super().__init__()

		self.epochs = epochs
		self.num_steps = 3 * epochs + (epochs % 2)  # Ensure complete cycles with even number of steps
		self.scale_factor = scale_factor
		self.base_lr = base_lr
		self.max_lr = max_lr
		self.updates_per_step = int(np.floor((num_samples * epochs) / (batch_size * self.num_steps)))
		if self.updates_per_step < 2:  # lr will not update properly
			self.updates_per_step = 2
		self.lr = base_lr
		self.diff = max_lr - base_lr
		self.change = self.diff / self.updates_per_step
		self.step = 0
		self.batch_count = 0
		self.iteration = 0
		self.history = {}

	def on_train_begin(self, logs={}):

		K.set_value(self.model.optimizer.lr, self.lr)

	def on_batch_end(self, batch, logs=None):

		logs = logs or {}
		self.iteration += 1

		self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
		self.history.setdefault('iteration', []).append(self.iteration)

		for k, v in logs.items():
			self.history.setdefault(k, []).append(v)

		if self.batch_count != self.updates_per_step:
			self.lr += self.change
			self.batch_count += 1
		elif self.step % 2 == 0:
			self.change = -self.change
			self.lr += self.change
			self.batch_count = 1
			self.step += 1
		else:
			self.diff *= self.scale_factor
			self.change = self.diff / self.updates_per_step  # lr at min and increases after even-numbered steps
			self.lr += self.change
			self.batch_count = 1  # Start of a new cycle
			self.step += 1

		K.set_value(self.model.optimizer.lr, self.lr)