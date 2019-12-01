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

sequence_length = params.sequence_length
sample_length = params.sample_max_length

encoder_layers = params.encoder_layers
encoder_hidden_size = params.encoder_hidden_size
decoder_layers = params.decoder_layers
decoder_hidden_size = params.decoder_hidden_size
embedding_size = params.embedding_size 

vocab_sizes = [len(dictionaries[0])] * 2

model = SequenceGenerator(dictionaries,
			  vocab_sizes,
			  encoder_layers,
			  encoder_hidden_size,
			  decoder_layers,
			  decoder_hidden_size,
			  embedding_size,
			  sequence_length)

checkpoint = tf.train.Checkpoint(optimizer=model.get_optimizer(),
				 encoder=model.get_encoder(),
				 decoder=model.get_decoder())

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))		

# taken from the training data to give the generator a warm start
seed = ['<start>', 0,157,2,158,2,158,2,158,2,119,133,149,
	157,2,120,134,150,158,2,120,134,150,
	158,2,120,134,150,158,2,120, '<end>']

sample = model.generate(seed, sample_length)
midi_util.sequence_to_midi(sample, sample_filepath, 'new_sample')

