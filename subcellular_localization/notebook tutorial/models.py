import numpy as np
import theano
import theano.tensor as T
import lasagne
from utils import LSTMAttentionDecodeFeedbackLayer, DropoutSeqPosLayer 

def FFN(batch_size, seq_len, n_hid, n_feat, n_class, lr, drop_prob):
	"""Compile a Feed-forward neural network for protein subcellular localization
	   
	Parameters:
		batch_size -- integer, minibatches size
		seq_len -- integer, sequence length
		n_hid -- integer, number of hidden neurons
		n_feat -- integer, number of features encoded
		n_class -- integer, number of classes to output
		lr -- float, learning rate
		drop_hid -- float, hidden neurons dropout

	Outputs:
		train_fn -- compiled theano function for training
		val_fn -- compiled theano function for validation/testing
		l_out -- output of the network, can be used to save the model parameters		
	"""

	# We use ftensor3 because the protein data is a 3D-matrix in float32 
	input_var = T.ftensor3('inputs')
	# ivector because the labels is a single dimensional vector of integers
	target_var = T.ivector('targets')


	# Input layer, holds the shape of the data
	l_in = lasagne.layers.InputLayer(shape=(batch_size, seq_len, n_feat), input_var=input_var, name='Input')

	# Dense layer with ReLu activation function
	l_dense = lasagne.layers.DenseLayer(l_in, num_units=n_hid*2, name="Dense",
	                                    nonlinearity=lasagne.nonlinearities.rectify)

	# Output layer with a Softmax activation function
	l_out = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_dense, p=drop_prob), num_units=n_class, 
                                  name="Softmax", nonlinearity=lasagne.nonlinearities.softmax)


	# Get output training, deterministic=False is used for training
	prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var}, deterministic=False)

	# Calculate the categorical cross entropy between the labels and the prediction
	t_loss = T.nnet.categorical_crossentropy(prediction, target_var)

	# Training loss
	loss = T.mean(t_loss)

	# Parameters
	params = lasagne.layers.get_all_params([l_out], trainable=True)

	# Get the network gradients and perform total norm constraint normalization
	all_grads = lasagne.updates.total_norm_constraint(T.grad(loss, params),3)

	# Update parameters using ADAM 
	updates = lasagne.updates.adam(all_grads, params, learning_rate=lr)


		# Get output validation, deterministic=True is only use for validation
	val_prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var}, deterministic=True)

	# Calculate the categorical cross entropy between the labels and the prediction
	t_val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var)

	# Validation loss 
	val_loss = T.mean(t_val_loss)



	train_fn = theano.function([input_var, target_var], [loss, prediction], updates=updates)
	val_fn = theano.function([input_var, target_var], [val_loss, val_prediction])

	return train_fn, val_fn, l_out

def CNN(batch_size, seq_len, n_hid, n_feat, n_class, n_filt, lr, drop_prob):
	"""Compile a Feed-forward neural network for protein subcellular localization
	   
	Parameters:
		batch_size -- integer, minibatches size
		n_hid -- integer, number of hidden neurons
		n_feat -- integer, number of features encoded
		n_class -- integer, number of classes to output
		lr -- float, learning rate
		drop_hid -- float, hidden neurons dropout

	Outputs:
		train_fn -- compiled theano function for training
		val_fn -- compiled theano function for validation/testing
		l_out -- output of the network, can be used to save the model parameters		
	"""
	
	# We use ftensor3 because the protein data is a 3D-matrix in float32 
	input_var = T.ftensor3('inputs')
	# ivector because the labels is a single dimensional vector of integers
	target_var = T.ivector('targets')


	# Input layer, holds the shape of the data
	l_in = lasagne.layers.InputLayer(shape=(batch_size, seq_len, n_feat), input_var=input_var, name='Input')

	# Shuffle shape to be properly read by the CNN layer
	l_shu = lasagne.layers.DimshuffleLayer(l_in, (0,2,1))

	# Convolutional layers with different filter size
	l_conv_a = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1, 
	                                      filter_size=3, nonlinearity=lasagne.nonlinearities.rectify)

	l_conv_b = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1, 
	                                      filter_size=5, nonlinearity=lasagne.nonlinearities.rectify)

	# The output is concatenated
	l_conc = lasagne.layers.ConcatLayer([l_conv_a, l_conv_b], axis=1)

	# Second CNN layer
	l_conv_final = lasagne.layers.Conv1DLayer(l_conc, num_filters=n_filt*2, pad='same', 
	                                          stride=1, filter_size=3, 
	                                          nonlinearity=lasagne.nonlinearities.rectify)

	# Max pooling is performed to downsample the input and reduce its dimensionality
	final_max_pool = lasagne.layers.MaxPool1DLayer(l_conv_final, 5)

	# Dense layer with ReLu activation function
	l_dense = lasagne.layers.DenseLayer(final_max_pool, num_units=n_hid*2, name="Dense",
	                                    nonlinearity=lasagne.nonlinearities.rectify)

	# Output layer with a Softmax activation function
	l_out = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_dense, p=drop_prob), num_units=n_class, name="Softmax", 
	                                  nonlinearity=lasagne.nonlinearities.softmax)



	# Get output training, deterministic=False is used for training
	prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var}, deterministic=False)

	# Calculate the categorical cross entropy between the labels and the prediction
	t_loss = T.nnet.categorical_crossentropy(prediction, target_var)

	# Training loss
	loss = T.mean(t_loss)

	# Parameters
	params = lasagne.layers.get_all_params([l_out], trainable=True)

	# Get the network gradients and perform total norm constraint normalization
	all_grads = lasagne.updates.total_norm_constraint(T.grad(loss, params),3)

	# Update parameters using ADAM 
	updates = lasagne.updates.adam(all_grads, params, learning_rate=lr)


		# Get output validation, deterministic=True is only use for validation
	val_prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var}, deterministic=True)

	# Calculate the categorical cross entropy between the labels and the prediction
	t_val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var)

	# Validation loss 
	val_loss = T.mean(t_val_loss)



	train_fn = theano.function([input_var, target_var], [loss, prediction], updates=updates)
	val_fn = theano.function([input_var, target_var], [val_loss, val_prediction])

	return train_fn, val_fn, l_out


def CNN_LSTM(batch_size, seq_len, n_hid, n_feat, n_class, n_filt, lr, drop_prob):
	"""Compile a Feed-forward neural network for protein subcellular localization
	   
	Parameters:
		batch_size -- integer, minibatches size
		n_hid -- integer, number of hidden neurons
		n_feat -- integer, number of features encoded
		n_class -- integer, number of classes to output
		lr -- float, learning rate
		drop_hid -- float, hidden neurons dropout

	Outputs:
		train_fn -- compiled theano function for training
		val_fn -- compiled theano function for validation/testing
		l_out -- output of the network, can be used to save the model parameters		
	"""
	
	# We use ftensor3 because the protein data is a 3D-matrix in float32 
	input_var = T.ftensor3('inputs')
	# ivector because the labels is a single dimensional vector of integers
	target_var = T.ivector('targets')
	# fmatrix because the masks to ignore the padded positions is a 2D-matrix in float32
	mask_var = T.fmatrix('masks')


	# Input layer, holds the shape of the data
	l_in = lasagne.layers.InputLayer(shape=(batch_size, None, n_feat), input_var=input_var, name='Input')

	# Mask input layer
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, None), input_var=mask_var, name='Mask')

	# Shuffle shape to be properly read by the CNN layer
	l_shu = lasagne.layers.DimshuffleLayer(l_in, (0,2,1))

	# Convolutional layers with different filter size
	l_conv_a = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1, 
	                                      filter_size=3, nonlinearity=lasagne.nonlinearities.rectify)

	l_conv_b = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1, 
	                                      filter_size=5, nonlinearity=lasagne.nonlinearities.rectify)

	# The output is concatenated
	l_conc = lasagne.layers.ConcatLayer([l_conv_a, l_conv_b], axis=1)

	# Second CNN layer
	l_conv_final = lasagne.layers.Conv1DLayer(l_conc, num_filters=n_filt*2, pad='same', 
	                                          stride=1, filter_size=3, 
	                                          nonlinearity=lasagne.nonlinearities.rectify)

	l_reshu = lasagne.layers.DimshuffleLayer(l_conv_final, (0,2,1))

	# Bidirectional LSTM layer, we only take the last hidden state (only_return_final)
	l_fwd = lasagne.layers.LSTMLayer(l_reshu, num_units=n_hid, name='LSTMFwd', mask_input=l_mask, 
	                                 only_return_final=True, nonlinearity=lasagne.nonlinearities.tanh)
	l_bck = lasagne.layers.LSTMLayer(l_reshu, num_units=n_hid, name='LSTMBck', mask_input=l_mask, 
	                                 only_return_final=True, backwards=True, nonlinearity=lasagne.nonlinearities.tanh)

	# Concatenate both layers
	l_conc_lstm = lasagne.layers.ConcatLayer([l_fwd, l_bck], axis=1)

	# Dense layer with ReLu activation function
	l_dense = lasagne.layers.DenseLayer(l_conc_lstm, num_units=n_hid*2, name="Dense",
	                                    nonlinearity=lasagne.nonlinearities.rectify)

	# Output layer with a Softmax activation function. Note that we include a dropout layer
	l_out = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_dense, p=drop_prob), num_units=n_class, name="Softmax", 
	                                  nonlinearity=lasagne.nonlinearities.softmax)


	# Get output training, deterministic=False is used for training
	prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var, l_mask: mask_var}, deterministic=False)

	# Calculate the categorical cross entropy between the labels and the prediction
	t_loss = T.nnet.categorical_crossentropy(prediction, target_var)

	# Training loss
	loss = T.mean(t_loss)

	# Parameters
	params = lasagne.layers.get_all_params([l_out], trainable=True)

	# Get the network gradients and perform total norm constraint normalization
	all_grads = lasagne.updates.total_norm_constraint(T.grad(loss, params),3)

	# Update parameters using ADAM 
	updates = lasagne.updates.adam(all_grads, params, learning_rate=lr)


	# Get output validation, deterministic=True is only use for validation
	val_prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var, l_mask: mask_var}, deterministic=True)

	# Calculate the categorical cross entropy between the labels and the prediction
	t_val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var)

	# Validation loss 
	val_loss = T.mean(t_val_loss)


	# Build functions
	train_fn = theano.function([input_var, target_var, mask_var], [loss, prediction], updates=updates)
	val_fn = theano.function([input_var, target_var, mask_var], [val_loss, val_prediction])

	return train_fn, val_fn, l_out


def CNN_LSTM_Att(batch_size, seq_len, n_hid, n_feat, n_class, n_filt, lr, drop_prob):
	"""Compile a Feed-forward neural network for protein subcellular localization
	   
	Parameters:
		batch_size -- integer, minibatches size
		n_hid -- integer, number of hidden neurons
		n_feat -- integer, number of features encoded
		n_class -- integer, number of classes to output
		lr -- float, learning rate
		drop_hid -- float, hidden neurons dropout

	Outputs:
		train_fn -- compiled theano function for training
		val_fn -- compiled theano function for validation/testing
		l_out -- output of the network, can be used to save the model parameters		
	"""
	
	# We use ftensor3 because the protein data is a 3D-matrix in float32 
	input_var = T.ftensor3('inputs')
	# ivector because the labels is a single dimensional vector of integers
	target_var = T.ivector('targets')
	# fmatrix because the masks to ignore the padded positions is a 2D-matrix in float32
	mask_var = T.fmatrix('masks')


	# Input layer, holds the shape of the data
	l_in = lasagne.layers.InputLayer(shape=(batch_size, None, n_feat), input_var=input_var, name='Input')

	# Mask input layer
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, None), input_var=mask_var, name='Mask')

	# Shuffle shape to be properly read by the CNN layer
	l_shu = lasagne.layers.DimshuffleLayer(l_in, (0,2,1))

	# Convolutional layers with different filter size
	l_conv_a = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1, 
	                                      filter_size=3, nonlinearity=lasagne.nonlinearities.rectify)

	l_conv_b = lasagne.layers.Conv1DLayer(l_shu, num_filters=n_filt, pad='same', stride=1, 
	                                      filter_size=5, nonlinearity=lasagne.nonlinearities.rectify)

	# The output is concatenated
	l_conc = lasagne.layers.ConcatLayer([l_conv_a, l_conv_b], axis=1)

	# Second CNN layer
	l_conv_final = lasagne.layers.Conv1DLayer(l_conc, num_filters=n_filt*2, pad='same', 
	                                          stride=1, filter_size=3, 
	                                          nonlinearity=lasagne.nonlinearities.rectify)

	l_reshu = lasagne.layers.DimshuffleLayer(l_conv_final, (0,2,1))

	l_fwd = lasagne.layers.LSTMLayer(l_reshu, num_units=n_hid, name='LSTMFwd', mask_input=l_mask,
	                                 nonlinearity=lasagne.nonlinearities.tanh)
	l_bck = lasagne.layers.LSTMLayer(l_reshu, num_units=n_hid, name='LSTMBck', mask_input=l_mask,
	                                 backwards=True, nonlinearity=lasagne.nonlinearities.tanh)

	# Concatenate both layers
	l_conc_lstm = lasagne.layers.ConcatLayer([l_fwd, l_bck], axis=2)


	l_att = LSTMAttentionDecodeFeedbackLayer(l_conc_lstm, mask_input=l_mask, 
	                                         num_units=n_hid*2, aln_num_units=n_hid, 
	                                         n_decodesteps=2, name='LSTMAttention')

	l_last_hid = lasagne.layers.SliceLayer(l_att, indices=-1, axis=1)

	# Dense layer with ReLu activation function
	l_dense = lasagne.layers.DenseLayer(l_last_hid, num_units=n_hid*2, name="Dense",
	                                    nonlinearity=lasagne.nonlinearities.rectify)

	# Output layer with a Softmax activation function. Note that we include a dropout layer
	l_out = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_dense, p=drop_prob), num_units=n_class, name="Softmax", 
	                                  nonlinearity=lasagne.nonlinearities.softmax)


	# Get output training, deterministic=False is used for training
	prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var, l_mask: mask_var}, deterministic=False)

	# Calculate the categorical cross entropy between the labels and the prediction
	t_loss = T.nnet.categorical_crossentropy(prediction, target_var)

	# Training loss
	loss = T.mean(t_loss)

	# Parameters
	params = lasagne.layers.get_all_params([l_out], trainable=True)

	# Get the network gradients and perform total norm constraint normalization
	all_grads = lasagne.updates.total_norm_constraint(T.grad(loss, params),3)

	# Update parameters using ADAM 
	updates = lasagne.updates.adam(all_grads, params, learning_rate=lr)


	# Get output validation, deterministic=True is only use for validation
	val_prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var, l_mask: mask_var}, deterministic=True)

	# Calculate the categorical cross entropy between the labels and the prediction
	t_val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var)

	# Validation loss 
	val_loss = T.mean(t_val_loss)


	# Build functions
	train_fn = theano.function([input_var, target_var, mask_var], [loss, prediction], updates=updates)
	val_fn = theano.function([input_var, target_var, mask_var], [val_loss, val_prediction, l_att.alpha])

	return train_fn, val_fn, l_out
