from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

class Trainer():
	
	def __init__(self, model, checkpoint_dir):
		self.model = model
		self.batch_size = model.get_batch_size()

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
		self.checkpoint = tf.train.Checkpoint(optimizer=self.model.get_optimizer(),
						      encoder=self.model.get_encoder(),
						      decoder=self.model.get_decoder())
		
	def prepare_datasets(self, inputs, targets, test_size = 0.2):
		(train_inputs, val_inputs,
		 train_targets, val_targets) = train_test_split(inputs, targets, test_size=test_size, shuffle = False)
		
		self.train_size = len(train_inputs) - len(train_inputs) % self.batch_size
		train_data = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets)) # set shuffle = false
		train_data = train_data.batch(self.batch_size, drop_remainder=True)
		
		self.val_size = len(val_inputs) - len(val_inputs) % self.batch_size
		val_data = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets))
		val_data = val_data.batch(self.batch_size, drop_remainder = True)
		
		self.train_data = train_data
		self.val_data = val_data
		
		
	def train_model(self, num_epochs):
		num_train_batches = self.train_size//self.batch_size
		num_val_batches = self.val_size//self.batch_size
		
		for epoch in range(num_epochs):
			print('Epoch {}'.format(epoch + 1))
			val_loss = 0
			val_accuracy = 0
			self.model.reset_encoder_hidden_states()
			for inputs, targets in self.val_data.take(num_val_batches):
				batch_loss, batch_accuracy = self.model.evaluate(inputs, targets)
				val_loss += batch_loss / targets.shape[1]
				val_accuracy += batch_accuracy
			val_loss /= num_val_batches
			val_accuracy /= num_val_batches


			train_loss = 0
			train_accuracy = 0
			self.model.reset_encoder_hidden_states()
			for (batch, (inputs, targets)) in enumerate(self.train_data.take(num_train_batches)):
				batch_loss, batch_accuracy = self.model.train_step(inputs, targets)
				train_loss += batch_loss / targets.shape[1]
				train_accuracy += batch_accuracy
				
				if batch % 100 == 0:
					print('\tBatch {} Loss: {:.4f}'.format(batch, batch_loss / targets.shape[1]))
			train_loss /= num_train_batches
			train_accuracy /= num_train_batches
			
			# saving (checkpoint) the model every 2 epochs
			if (epoch + 1) % 2 == 0:
				self.checkpoint.save(file_prefix = self.checkpoint_prefix)

			print('----------------------------------------------------------')
			print('Epoch {} Training Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, train_loss, 100*train_accuracy))
			print('Epoch {} Validation Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, val_loss, 100*val_accuracy))
			print('----------------------------------------------------------')
