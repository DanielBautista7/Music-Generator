''' Model hyperparameters '''

num_epochs = 100
batch_size = 64
sequence_length = 32

encoder_layers = 2
encoder_hidden_size = 256

decoder_layers = 2
decoder_hidden_size = 256

embedding_size = 64

dropout = 0.2
# max clipping value for gradient norm
max_grad_norm = 5.

# portion of data to be used for the test set
test_size = 0.3

# maximum length of the sampled musical sequence
sample_max_length = 250

checkpoint_dir = './checkpoints'
generated_sample_filepath = './samples'
