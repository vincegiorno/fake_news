from keras.callbacks import LambdaCallback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

class LearningRateFinder:
	def __init__(self, model, stopFactor=4, beta=0.98):
		self.model = model
		self.stopFactor = stopFactor
		self.beta = beta  # Used in computing smoothed average loss

		# initialize learning rate and loss lists, local variables
		self.lrs = []
		self.losses = []
		self.lr_mult = 1
		self.avg_loss = 0
		self.best_loss = 1e9
		self.batch_num = 0
		self.weightsFile = None

	def reset(self):
		# re-initialize all variables from our constructor
		self.lrs = []
		self.losses = []
		self.lr_mult = 1
		self.avg_loss = 0
		self.best_loss = 1e9
		self.batch_num = 0

	def on_batch_end(self, batch, logs):
		lr = K.get_value(self.model.optimizer.lr)
		self.lrs.append(lr)

		self.batch_num += 1
		self.avg_loss = (self.beta * self.avg_loss) + ((1 - self.beta) * logs['loss'])
		smooth = self.avg_loss / (1 - (self.beta ** self.batch_num))
		self.losses.append(smooth)

		stop_loss = self.stopFactor * self.best_loss

		if self.batch_num > 1 and smooth > stop_loss:
			self.model.stop_training = True
			return

		if self.batch_num == 1 or smooth < self.best_loss:
			self.best_loss = smooth

		lr *= self.lr_mult
		K.set_value(self.model.optimizer.lr, lr)

	def find(self, train_x, train_y, epochs, batch_size, start_lr=1e-8, end_lr=1., verbose=1):

		self.reset()  # Reset class-specific variables

		steps_per_epoch = np.ceil(len(train_x) / float(batch_size))
		num_batch_updates = epochs * steps_per_epoch

		# Calculate multiplier for updating learning rate on a logarithmic scale
		self.lr_mult = (end_lr / start_lr) ** (1.0 / num_batch_updates)

		K.set_value(self.model.optimizer.lr, start_lr)

		# construct a Keras callback to invoke the custom on_batch_end function
		callback = LambdaCallback(on_batch_end=lambda batch, logs:
			self.on_batch_end(batch, logs))

		self.model.fit(
			train_x, train_y,
			batch_size=batch_size,
			epochs=epochs,
			callbacks=[callback],
			verbose=verbose)
		
	def plot_loss(self, skip_begin=10, skip_end=1):
		# grab the learning rate and losses values to plot
		lrs = self.lrs[skip_begin:-skip_end]
		losses = self.losses[skip_begin:-skip_end]

		# plot the learning rate vs. loss
		plt.figure(figsize=(10, 5))
		plt.plot(lrs, losses)
		plt.xscale("log")
		plt.xlabel("Learning Rate (Log Scale)")
		plt.ylabel("Loss")
		plt.title("Loss vs learning rate", size=16)
		plt.tick_params(axis='both', which='minor', bottom='off', left='off')
		plt.show()