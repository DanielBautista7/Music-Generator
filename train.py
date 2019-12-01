import tensorflow as tf
import hyperparameters as params
from trainer import Trainer
from model import SequenceGenerator
from load_data import inputs, targets, dictionaries

num_epochs = params.num_epochs
batch_size = params.batch_size
sequence_length = params.sequence_length

dropout = params.dropout
max_grad_norm = params.max_grad_norm

test_size = params.test_size

embedding_size = params.embedding_size
encoder_layers = params.encoder_layers
encoder_hidden_size = params.encoder_hidden_size
decoder_layers = params.decoder_layers
decoder_hidden_size = params.decoder_hidden_size

checkpoint_dir = params.checkpoint_dir

vocab_sizes = [len(dictionaries[0])] * 2

model = SequenceGenerator(dictionaries,
			  vocab_sizes,
			  encoder_layers,
			  encoder_hidden_size,
			  decoder_layers,
			  decoder_hidden_size,
			  embedding_size,
			  sequence_length,
			  dropout,
			  max_grad_norm,
			  batch_size)

trainer = Trainer(model, checkpoint_dir)

trainer.prepare_datasets(inputs, targets, test_size)
trainer.train_model(num_epochs)

