import hyperparameters as params
import numpy as np
import collections
import glob
import os

#----------------------- To be determined while preprocessing inputs
batch_size = params.batch_size
sequence_length = params.sequence_length
#------------------------------------------------------ Dictionaries
symbol_counts = collections.Counter()
symbol_ix_dict = {}
ix_symbol_dict = {}
#-------------------------------------------------------------------


print("Preparing Input...")

dirPath = os.path.dirname(os.path.realpath(__file__))
inputFolder = os.path.join(dirPath,"data")

print("Input Folder: "+inputFolder)

inputFiles = os.path.join(inputFolder,"*.csv")

print("Looking for files matching: "+inputFiles)
seqFileList = glob.glob(inputFiles,recursive = True)
print("Number of files found: %d" % len(seqFileList))

dataVectorLength = 0

sequenceCount = 0
print("Loading Data...")
data_vector = np.array([], dtype=int)

for seqFile in seqFileList:
	seq = np.loadtxt(seqFile, delimiter = ',').astype('int32')
	data_vector = np.concatenate((data_vector, seq))
	filename = seqFile.rsplit("/",1)[-1]
	print("\tsuccessfully loaded: "+filename)
	
symbol_counts.update(data_vector)
symbol_counts.update({'<start>':1})
symbol_counts.update({'<end>':1})

for ix, symbol in enumerate(np.sort(list(set(symbol_counts)))):	
	symbol_ix_dict[symbol] = ix
	ix_symbol_dict[ix] = symbol

symbol_to_ix = np.vectorize(lambda x : symbol_ix_dict[str(x)])
data_vector = symbol_to_ix(data_vector)


print('Defining inputs and targets...')
print('Untruncated length: %d' % len(data_vector))
unused_tokens = len(data_vector) % ((sequence_length - 2) * batch_size)
data_vector = data_vector[: - unused_tokens].reshape((-1, (sequence_length - 2)))
print('Truncated length: %d' % data_vector.size)
print('Unused tokens: %d' % unused_tokens)

num_points = data_vector.shape[0]
start_tokens = np.expand_dims([symbol_ix_dict['<start>']] * num_points, 1)
end_tokens = np.expand_dims([symbol_ix_dict['<end>']] * num_points, 1)

data_vector = np.concatenate((start_tokens, data_vector, end_tokens), axis=1)

inputs = data_vector[:-1]
targets = data_vector[1:]
print('Successfully prepared inputs and targets\n')
dictionaries = (symbol_ix_dict, ix_symbol_dict)
