import os
import sys
import numpy as np
import theano
import theano.tensor as T
import time
import lasagne
from metrics_mc import *
from model import neural_network
from confusionmatrix import ConfusionMatrix
from utils import iterate_minibatches

# Input options
n_class = 10
batch_size = 128
seq_len = 1000
n_hid = 256
lr = 0.0005
num_epochs = 200
drop_per = 0.2
drop_hid = 0.5
n_filt = 10

# Load data
print "Loading data...\n"
test_data = np.load("data/test.npz")
train_data = np.load("data/train.npz")

# Test set
X_test = test_data['X_test']
y_test = test_data['y_test']
mask_test = test_data['mask_test']

# Output vectors from test set
complete_alpha = np.zeros((X_test.shape[0],seq_len))
complete_context = np.zeros((X_test.shape[0],n_hid*2))
complete_test = np.zeros((X_test.shape[0],n_class))

# Training set
X_train = train_data['X_train']
y_train = train_data['y_train']
mask_train = train_data['mask_train']
partition = train_data['partition']

# Number of features
n_feat = np.shape(X_test)[2]

# Training
for i in range(1,5):
	# Network compilation
	print("Compilation model {}\n".format(i))
	train_fn, val_fn, network_out = neural_network(batch_size, seq_len, n_hid, n_feat, n_class, lr, drop_per, drop_hid, n_filt)
	
	# Train and validation sets
	train_index = np.where(partition != i)
	val_index = np.where(partition == i)
	X_tr = X_train[train_index].astype(np.float32)
	X_val = X_train[val_index].astype(np.float32)
	y_tr = y_train[train_index].astype(np.int32)
	y_val = y_train[val_index].astype(np.int32)
	mask_tr = mask_train[train_index].astype(np.float32)
	mask_val = mask_train[val_index].astype(np.float32)

	print("Validation shape: {}".format(X_val.shape))
	print("Training shape: {}".format(X_tr.shape))
	
	eps = []
	best_val_acc = 0

	print "Start training\n"	
	for epoch in range(num_epochs):
	    # Calculate epoch time
	    start_time = time.time()
	    
	    # Full pass training set
	    train_err = 0
	    train_batches = 0
	    confusion_train = ConfusionMatrix(n_class)
	    
	    # Generate minibatches and train on each one of them	
	    for batch in iterate_minibatches(X_tr, y_tr, mask_tr, batch_size, shuffle=True):
		inputs, targets, in_masks = batch
		tr_err, predict = train_fn(inputs, targets, in_masks)
		train_err += tr_err
		train_batches += 1
		preds = np.argmax(predict, axis=-1)
		confusion_train.batch_add(targets, preds)
	    
	    train_loss = train_err / train_batches
	    train_accuracy = confusion_train.accuracy()
	    cf_train = confusion_train.ret_mat()	    

		
	    # Full pass validation set
	    val_err = 0
	    val_batches = 0
	    confusion_valid = ConfusionMatrix(n_class)
	    
	    # Generate minibatches and train on each one of them	
	    for batch in iterate_minibatches(X_val, y_val, mask_val, batch_size):
		inputs, targets, in_masks = batch
		err, predict_val, alpha, context = val_fn(inputs, targets, in_masks)
		val_err += err
		val_batches += 1
		preds = np.argmax(predict_val, axis=-1)
		confusion_valid.batch_add(targets, preds)
		
	    val_loss = val_err / val_batches
	    val_accuracy = confusion_valid.accuracy()
	    cf_val = confusion_valid.ret_mat()
            
	    f_val_acc = val_accuracy

	    # Full pass test set if validation accuracy is higher
	    if f_val_acc >= best_val_acc:
		
	    	test_batches = 0
		# Matrices to store all output information
		test_alpha = np.array([], dtype=np.float32).reshape(0,seq_len)
		test_context = np.array([], dtype=np.float32).reshape(0,n_hid*2)
		test_pred = np.array([], dtype=np.float32).reshape(0,n_class)
		
		for batch in iterate_minibatches(X_test, y_test, mask_test, batch_size, shuffle=False, test=True):
			inputs, targets, in_masks = batch
			err, net_out, alpha, context = val_fn(inputs, targets, in_masks)
			
			test_batches += 1	
			last_alpha = alpha[:,-1:,:].reshape((batch_size, seq_len))
			test_alpha = np.concatenate((test_alpha, last_alpha), axis=0)
			test_context = np.concatenate((test_context, context), axis=0)
			test_pred = np.concatenate((test_pred, net_out),axis=0)		

		best_val_acc = f_val_acc
	    
	    eps += [epoch]
	    
	    # Then we print the results for this epoch:
	    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
	    print confusion_valid
	    print("  training loss:\t\t{:.6f}".format(train_loss))
	    print("  validation loss:\t\t{:.6f}".format(val_loss))
	    print("  training accuracy:\t\t{:.2f} %".format(train_accuracy * 100))
	    print("  validation accuracy:\t\t{:.2f} %".format(val_accuracy * 100))
	    print("  training Gorodkin:\t\t{:.2f}".format(gorodkin(cf_train)))
	    print("  validation Gorodkin:\t\t{:.2f}".format(gorodkin(cf_val)))
	    print("  training IC:\t\t{:.2f}".format(IC(cf_train)))
	    print("  validation IC:\t\t{:.2f}".format(IC(cf_val)))

	# Output matrices test set are summed at the end of each training
	complete_test += test_pred[:X_test.shape[0]]
	complete_context += test_context[:X_test.shape[0]]
	complete_alpha += test_alpha[:X_test.shape[0]]


# The test output from the 4 trainings is averaged
test_softmax = complete_test / 4.0
context_vectors = complete_context / 4.0
alpha_weight = complete_alpha / 4.0

# Final test accuracy and confusion matrix
confusion_test = ConfusionMatrix(n_class)
loc_pred = np.argmax(test_softmax, axis=-1)
confusion_test.batch_add(y_test, loc_pred)
test_accuracy = confusion_test.accuracy()
cf_test = confusion_test.ret_mat()

print "FINAL TEST RESULTS"
print confusion_test
print("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
print("  test Gorodkin:\t\t{:.2f}".format(gorodkin(cf_test)))
print("  test IC:\t\t{:.2f}".format(IC(cf_test)))
