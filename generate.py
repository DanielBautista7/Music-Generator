from load_data import dictionaries
from model import SequenceGenerator, Encoder
import hyperparameters as params
import tensorflow as tf
import midi_util
import os

sample_filepath = params.generated_sample_filepath
checkpoint_dir = params.checkpoint_dir

if not os.path.exists(sample_filepath):
	os.makedirs(sample_filepath)

batch_size = params.batch_size
sequence_length = params.sequence_length
sample_length = params.sample_max_length

embedding_size = params.embedding_size 
encoder_hidden_size = params.encoder_hidden_size
decoder_hidden_size = params.decoder_hidden_size


vocab_sizes = [len(dictionaries[0])] * 2

model = SequenceGenerator(dictionaries,
			  vocab_sizes,
			  encoder_hidden_size,
			  decoder_hidden_size,
			  embedding_size,
			  sequence_length,
			  batch_size)

encoder = Encoder(encoder_hidden_size, embedding_size, len(dictionaries[0]), sequence_length, batch_size = 1)
checkpoint = tf.train.Checkpoint(optimizer=model.get_optimizer(),
				 encoder=model.encoder,
				 decoder=model.get_decoder())

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))		

# taken from the training data to give the generator a warm start
seed = ['<start>', 0,157,2,158,2,158,2,158,2,119,133,149,
	157,2,120,134,150,158,2,120,134,150,
	158,2,120,134,150,158,2,120, '<end>']

sample = model.generate(encoder, seed, sample_length)

midi_util.sequence_to_midi(sample, sample_filepath, 'new_sample')

