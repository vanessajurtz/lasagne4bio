import numpy as np
import theano
import theano.tensor as T
import lasagne
from utils import LSTMAttentionDecodeFeedbackLayer, DropoutSeqPosLayer 

def neural_network(batch_size, n_hid, n_feat, n_class, lr, drop_per, drop_hid, n_filt):
	"""Compile a Convolutional BLSTM neural network for protein subcellular localization
	   
	Parameters:
		batch_size -- integer, minibatches size
		n_hid -- integer, number of hidden neurons
		n_feat -- integer, number of features encoded
		n_class -- integer, number of classes to output
		lr -- float, learning rate
		drop_per -- float, input dropout
		drop_hid -- float, hidden neurons dropout
		n_filt -- integer, number of filter in the first convolutional layer

	Outputs:
		train_fn -- compiled theano function for training
		val_fn -- compiled theano function for validation/testing
		l_out -- output of the network, can be used to save the model parameters		
	"""
	# Prepare Theano variables for inputs, masks and targets
	w_inits = lasagne.init.Orthogonal('relu')
	input_var = T.tensor3('inputs')
	target_var = T.ivector('targets')
	mask_var = T.matrix('masks')
	
	# Input layer, holds the shape of the data
	l_in = lasagne.layers.InputLayer(shape=(batch_size, None, n_feat), input_var=input_var)

	# Dropout positions of the protein sequence	
	l_indrop = DropoutSeqPosLayer(l_in, p=drop_per)
	
	# Input layer with masks
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, None), input_var=mask_var)
	
	#### CNN ####
	# Size of convolutional layers
	f_size_a = 1
	f_size_b = 3
	f_size_c = 5
	f_size_d = 9
	f_size_e = 15
	f_size_f = 21
	
	# Shuffle shape to be properly read by the CNN layer
	l_shu = lasagne.layers.DimshuffleLayer(l_indrop, (0,2,1))
	
	# First convolutional layers
	l_conv_a = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1, W=w_inits, filter_size=f_size_a, nonlinearity=lasagne.nonlinearities.rectify)
	l_conv_b = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1,W=w_inits, filter_size=f_size_b, nonlinearity=lasagne.nonlinearities.rectify)
	l_conv_c = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1,W=w_inits, filter_size=f_size_c, nonlinearity=lasagne.nonlinearities.rectify)
	l_conv_d = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1,W=w_inits, filter_size=f_size_d, nonlinearity=lasagne.nonlinearities.rectify)
	l_conv_e = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1,W=w_inits, filter_size=f_size_e, nonlinearity=lasagne.nonlinearities.rectify)
	l_conv_f = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1,W=w_inits, filter_size=f_size_f, nonlinearity=lasagne.nonlinearities.rectify)
	
	# Concatenate all CNN layers
	l_conc = lasagne.layers.ConcatLayer([l_conv_a, l_conv_b, l_conv_c, l_conv_d, l_conv_e, l_conv_f], axis=1)
	
	# Second CNN layer
	l_conv_final = lasagne.layers.Conv1DLayer(l_conc, num_filters=64, pad='same', stride=1, filter_size=f_size_b, nonlinearity=lasagne.nonlinearities.rectify)
	
	# Reshuffle to initial shape
	l_indrop = lasagne.layers.DimshuffleLayer(l_conv_final, (0,2,1))
	
	# Dropout
	l_indrop = lasagne.layers.dropout(l_indrop, p=drop_hid)
	
	#### Attention mechanism ####
	# LSTM forward and backward layers
	l_fwd = lasagne.layers.LSTMLayer(l_indrop, num_units=n_hid, name='LSTMFwd', mask_input=l_mask, cell_init=lasagne.init.Orthogonal(), hid_init=lasagne.init.Orthogonal(), nonlinearity=lasagne.nonlinearities.tanh, grad_clipping=2)
	l_bck = lasagne.layers.LSTMLayer(l_indrop, num_units=n_hid, name='LSTMBck', mask_input=l_mask, cell_init=lasagne.init.Orthogonal(), hid_init=lasagne.init.Orthogonal(),	backwards=True, nonlinearity=lasagne.nonlinearities.tanh, grad_clipping=2)
	
	# Concatenate both layers
	l_conc_lstm = lasagne.layers.ConcatLayer([l_fwd, l_bck], axis=2)
	
	# Attention mechanism
	l_dec = LSTMAttentionDecodeFeedbackLayer(l_conc_lstm, mask_input=l_mask, num_units=n_hid*2, aln_num_units=n_hid, n_decodesteps=10, name='LSTMAttention')
	
	# Slice last layer of the attention mechanism (context vector)
	l_last_hid = lasagne.layers.SliceLayer(l_dec, indices=-1, axis=1)

	# Final fully connected dense layer	
	l_dense = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_last_hid, p=drop_hid), name="Dense", num_units=n_hid*2, W=w_inits, nonlinearity=lasagne.nonlinearities.rectify)
	
	# Softmax output layer
	l_out = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_dense, p=drop_hid), num_units=n_class, name="Softmax", W=w_inits, nonlinearity=lasagne.nonlinearities.softmax)
	
	# Get output training
	prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var, l_mask: mask_var}, deterministic=False)

	# Loss function
	t_loss = T.nnet.categorical_crossentropy(prediction, target_var)
	loss = T.mean(t_loss)

	# Parameters
	params = lasagne.layers.get_all_params([l_out], trainable=True)
	
	all_grads = lasagne.updates.total_norm_constraint(T.grad(loss, params),3)
	
	# Update using ADAM
	updates = lasagne.updates.adam(all_grads, params, learning_rate=lr)	
	
	# Get output validation
	test_prediction, context_vec = lasagne.layers.get_output([l_out, l_last_hid], inputs={l_in: input_var, l_mask: mask_var}, deterministic=True)
	
	# Loss function 
	t_test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
	test_loss = T.mean(t_test_loss)
	
	# Build functions
	train_fn = theano.function([input_var, target_var, mask_var], [loss, prediction], updates=updates)
	val_fn = theano.function([input_var, target_var, mask_var], [test_loss, test_prediction, l_dec.alpha, context_vec])	
	return train_fn, val_fn, l_out
