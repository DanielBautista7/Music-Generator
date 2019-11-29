''' Model hyperparameters '''

num_epochs = 30
batch_size = 64
sequence_length = 32

embedding_size = 32 
encoder_hidden_size = 128
decoder_hidden_size = 128

# portion of data to be used for the test set
test_size = 0.3

# maximum length of the sampled musical sequence
sample_max_length = 250

checkpoint_dir = './checkpoints'
generated_sample_filepath = './samples'
