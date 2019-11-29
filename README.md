Background
==========
In my undergraduate thesis I explored the ability of a deep LSTM model, optimized with an embedding layer and regularized with dropout as well as recurrent batch normalization, in generating music similar in style to the Bach Partitas for Violin.
This was done using a data set of the Bach Partitas which I constructed using a sequential encoding scheme proposed in prior work in the field of Algorithmic Composition.
The full thesis paper can be accessed [here](https://drive.google.com/file/d/19fhuEagvPUEYrENxCyYTTJtyfnv7HAd5/view?usp=sharing).

This personal project investigates the effectiveness of an alternative model for the same task. The model used is a GRU encoder-decoder with Bahdanau attention. Note that this is a work in progress and is currently a prototype. Further optimizations and regularizations have yet to be implemented.

Required libraries: Tensorflow 2.0, keras, sklearn, numpy, and the [python-midi](https://github.com/vishnubob/python-midi) library by [vishnubob](https://github.com/vishnubob).


The **_data_** folder contains musical data for the Bach Partitas encoded using the scheme detailed in
pages 17 to 19 of my [undergraduate thesis](https://drive.google.com/file/d/19fhuEagvPUEYrENxCyYTTJtyfnv7HAd5/view?usp=sharing).

Training the Model
=========
The hyperparameters of the model can be modified in **_hyperparameters.py_** and the model can be trained by running **_train.py_**. This creates a **_checkpoints_** folder where model checkpoints are saved.

Generating Music
=========
The last saved model can be used for sampling music by running **_generate.py_**. This creates a **_samples_** folder where MIDI files of sampled music are saved after being converted from musical sequence data using **_midi_util.py_**.

