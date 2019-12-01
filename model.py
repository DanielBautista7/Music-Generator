from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
import numpy as np

''' GRU Encoder-Decoder model with Bahdanau attention used as a sequence generator '''

class SequenceGenerator():

	def __init__(self, dictionaries, vocab_sizes, encoder_layers, encoder_units,  decoder_layers,
			decoder_units, embedding_size, seq_length, dropout = 0., max_grad_norm = 5., batch_size = 1):
		self.symbol_ix_dict = dictionaries[0]
		self.ix_symbol_dict = dictionaries[1]
		
		self.encoder_layers = encoder_layers
		self.encoder_units = encoder_units

		self.decoder_layers = decoder_layers
		self.decoder_units = decoder_units
		
		self.embedding_size = embedding_size
		self.dropout = dropout
		
		self.input_vocab_size = vocab_sizes[0]
		self.target_vocab_size = vocab_sizes[1]

		self.seq_length = seq_length
		self.batch_size = batch_size
		
		self.optimizer = tf.keras.optimizers.Adam(clipnorm = max_grad_norm)

		self.encoder = Encoder(self.encoder_layers,
				       self.encoder_units,
				       self.embedding_size,
				       self.input_vocab_size,
				       self.seq_length,
				       self.dropout,
				       self.batch_size)
		
		self.decoder = Decoder(self.decoder_layers,
				       self.decoder_units,
				       self.embedding_size,
				       self.target_vocab_size,
				       self.dropout,
				       self.batch_size)

		self.loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
		
	
	def evaluate(self, inputs, targets, training = True):
		loss = 0
		enc_output, enc_hidden_state = self.encoder(inputs, training = training)
		dec_hidden_state = enc_hidden_state
		
		dec_input = tf.expand_dims([self.symbol_ix_dict['<start>']] * self.batch_size, 1)
		
		predictions = np.empty((targets.shape[0], 1))
		for t in range(1, targets.shape[1]):
			logits, dec_hidden_state, _ = self.decoder(dec_input, dec_hidden_state, enc_output, training = training)
			loss += self.loss_function(targets[:, t], logits)
			dec_input = tf.expand_dims(targets[:, t], 1)
			
			probabilities = tf.math.softmax(logits, 1).numpy()
			cur_predictions = np.expand_dims(np.argmax(probabilities, 1), 1)
			predictions = np.hstack((predictions, cur_predictions))
		
		accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, tf.cast(predictions, tf.int64)), tf.float32)).numpy()
		return loss, accuracy
		
	def train_step(self, inputs, targets):
		with tf.GradientTape() as tape:
			loss, accuracy = self.evaluate(inputs, targets)
			variables = self.encoder.trainable_variables + self.decoder.trainable_variables
			gradients = tape.gradient(loss, variables)
			self.optimizer.apply_gradients(zip(gradients, variables))
		return loss, accuracy
		
		
	def loss_function(self, targets, logits):
		# add small constant for numerical stability
		loss = self.loss_object(targets, logits + 1e-7) 
		
		mask = tf.math.logical_not(tf.math.equal(targets, 0))
		mask = tf.cast(mask, dtype=loss.dtype)

		loss *= mask
		return tf.reduce_mean(loss)
	
	def generate(self, seed, max_length = 1e4):
		print('Sampling new sequence...')
		assert len(seed) == self.seq_length, 'Seed must have %d tokens' % self.seq_length
		sample = seed[1:-1]

		start_token = self.symbol_ix_dict['<start>']
		end_token = self.symbol_ix_dict['<end>']

		# In the encoding scheme used, 1 signifies end of composition
		end_of_sequence = self.symbol_ix_dict['1']
		
		enc_input = tf.expand_dims(self.symbol_to_ix(seed), 0)
		self.reset_encoder_hidden_states()
		while not sample[-1] == end_of_sequence and len(sample) < max_length:
			enc_output, enc_hidden_state = self.encoder(enc_input, training = False)
			dec_hidden_state = enc_hidden_state
			
			enc_input = [start_token]
			dec_input = tf.expand_dims([start_token], 0)
			prediction = None
			while len(enc_input) < self.seq_length and not prediction == end_of_sequence:
				logits, dec_hidden_state, _ = self.decoder(dec_input, dec_hidden_state, enc_output, training = False)				
				probabilities = tf.math.softmax(logits, 1).numpy()
				prediction = np.argmax(probabilities, 1)[0]
				
				if prediction == end_token:
					break
				
				enc_input += [prediction]
				dec_input = tf.expand_dims([prediction], 0)				
				sample.append(int(self.ix_symbol_dict[prediction]))
			
			if len(enc_input) < self.seq_length:
				enc_input.append(end_token)
			else:
				enc_input[-1] = end_token

			if not prediction == end_of_sequence and len(enc_input) < self.seq_length:
				sample.append(int(self.ix_symbol_dict[end_of_sequence]))
				break
			enc_input = tf.expand_dims(enc_input, 0)
		print('Sampling completed.\nSample length: %d' % len(sample))
		return sample
				
	
	def reset_encoder_hidden_states(self):
		return self.encoder.reset_hidden_states()

	def reset_dropout_masks(self):
		self.encoder.reset_dropout_masks()
		self.decoder.reset_dropout_masks()
	
	def get_optimizer(self):
		return self.optimizer
	
	def get_encoder(self):
		return self.encoder
	
	def get_decoder(self):
		return self.decoder
	
	def get_batch_size(self):
		return self.batch_size
	
	def symbol_to_ix(self, sequence):
		# TODO handle case when element is not in dictionary
		return [self.symbol_ix_dict[str(x)] for x in sequence]
		
	def ix_to_symbol(self, sequence):
		# TODO handle case when element is not in dictionary
		return [self.ix_symbol_dict[int(x)] for x in sequence]


class Encoder(tf.keras.Model):
	def __init__(self, num_layers, hidden_units, embedding_size, vocab_size, seq_length, dropout, batch_size):
		super(Encoder, self).__init__()
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.hidden_units = hidden_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
		self.gru_layers = [tf.keras.layers.GRU(self.hidden_units,
						       activation = 'relu',
						       dropout = dropout,
						       stateful = True,
						       return_sequences=True,
						       return_state=True,
						       recurrent_initializer='glorot_uniform')
				   for _ in range(num_layers)]

		self.batch_norm_layers = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
		
		self.gru_layers[0].build((batch_size, seq_length, embedding_size))
		for gru in self.gru_layers[1:]:
			gru.build((batch_size, seq_length, hidden_units))

	def call(self, x, training = True):
		x = self.embedding(x)
		layer_output, layer_state = self.gru_layers[0](x, training = training) 
		layer_output = self.batch_norm_layers[0](layer_output)
		for gru, batch_norm in zip(self.gru_layers[1:], self.batch_norm_layers[1:]):
			layer_output, layer_state = gru(layer_output,
							initial_state = layer_state,
							training = training)
			layer_output = batch_norm(layer_output)
		
		return layer_output, layer_state

	def reset_hidden_states(self):
		for gru in self.gru_layers:
			gru.reset_states()
	
	def reset_dropout_masks(self):
		for gru in self.gru_layers:
			gru.reset_dropout_mask()

		
class Decoder(tf.keras.Model):
	def __init__(self, num_layers, hidden_units, embedding_size, vocab_size, dropout, batch_size):
		super(Decoder, self).__init__()
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.hidden_units = hidden_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
		self.gru_layers = [tf.keras.layers.GRU(self.hidden_units,
						       activation = 'linear' if i == num_layers - 1 else 'relu',
						       dropout = dropout,
						       return_sequences=True,
						       return_state=True,
						       recurrent_initializer='glorot_uniform')
				   for i in range(num_layers)]
		
		self.batch_norm_layers = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

		self.attention = BahdanauAttention(self.hidden_units)
	
	def call(self, x, hidden_state, enc_output, training = True):
		context_vector, attention_weights = self.attention(hidden_state, enc_output)
		x = self.embedding(x)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
		
		layer_output, layer_state = self.gru_layers[0](x, training = training)
		layer_output = self.batch_norm_layers[0](layer_output)
		
		for gru, batch_norm in zip(self.gru_layers[1:], self.batch_norm_layers[1:]):
			layer_output, layer_state = gru(layer_output,
							initial_state = layer_state,
							training = training)
			layer_output = batch_norm(layer_output)

		layer_output = tf.reshape(layer_output, (-1, layer_output.shape[2]))
		return layer_output, layer_state, attention_weights

	def reset_dropout_masks(self):
		for gru in self.gru_layers:
			gru.reset_dropout_mask()

		
class BahdanauAttention(tf.keras.layers.Layer):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, query, values):
		hidden_with_time_axis = tf.expand_dims(query, 1)

		score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

		attention_weights = tf.nn.softmax(score, axis=1)

		context_vector = attention_weights * values
		context_vector = tf.reduce_sum(context_vector, axis=1)

		return context_vector, attention_weights

