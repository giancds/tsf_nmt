These are my experiments re-implementing "Neural Machine Translation by Jointly Learning to Align 
and Translate " paper by [Bahdanau et al. (2014)](http://arxiv.org/abs/1409.0473) and integrating 
the techniques described in  "On Using Very Large Target Vocabulary for Neural Machine Translation" 
by [Jean et al. (2015)](http://arxiv.org/abs/1412.2007) using TensorFlow. 

I'm heavily relying on Tensorflow's seq2seq interfaces to construct the model while adding the 
bidirectional encoder.

I'm following the PEP8 conventions for coding with one change: I'm using lines 100 characters long. 
(Sorry python purists). 

**This is a work in progress and currently cannot be used to train any models.**

###### Dependencies (so far):

* [TensorFlow](http://tensorflow.org/)