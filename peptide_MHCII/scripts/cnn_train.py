#!/usr/bin/env python

"""
Convolutional net training using the Lasagne library:
https://github.com/Lasagne
"""

from __future__ import print_function
import argparse
import sys
import os
import time

import numpy as np
from scipy.io import netcdf
import theano
import theano.tensor as T

import lasagne

import data_io_func
import NN_func

theano.config.floatX='float32'


##########################################################################
#	FUNCTIONS
##########################################################################



############################### Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.


def iterate_minibatches(pep, mhc, targets, batchsize):
    assert pep.shape[0] == mhc.shape[0] == targets.shape[0]
    # shuffle:
    indices = np.arange(len(pep))
    np.random.shuffle(indices)
    for start_idx in range(0, len(pep) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield pep[excerpt],mhc[excerpt],targets[excerpt]


################################################################################
#	PARSE COMMANDLINE OPTIONS
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-training_data', '--trainfile',  help="file with training data")
parser.add_argument('-validation_data', '--valfile',  help="file with validation data (for early stopping)")
parser.add_argument('-model_out', '--modelfile',  help="file to store best model", default='model.npz')
parser.add_argument('-max_pep_seq_length', '--max_pep_seq_length',  help="maximal peptide sequence length, default = 15", default=15)
parser.add_argument('-motif_length', '--motif_length',  help="motif length, default = 9", default=9)
parser.add_argument('-batch_size', '--batch_size',  help="Mini batch size, default = 20", default=20)
parser.add_argument('-epochs', '--epochs',  help="Number of training epochs, default = 30", default=30)
parser.add_argument('-n_filters', '--n_filters',  help="Number of filters, default = 20", default=20)
parser.add_argument('-learning_rate', '--learning_rate',  help="Learning rate, default = 0.01", default=0.01)
parser.add_argument('-update', '--update',  help="Method for weight update, default = sgd", default="sgd")
parser.add_argument('-activation', '--activation',  help="Activation function, default = sigmoid", default="sigmoid")
parser.add_argument('-dropout', '--dropout',  help="Dropout (0-0.99 corresponding to 0-99%), default = 0", default=0)
parser.add_argument('-n_hid', '--n_hid',  help="Size of 1st hidden layer, default = 60", default=60)
parser.add_argument('-architecture', '--architecture',  help="Neural network architecture, default = cnn", default="cnn")
parser.add_argument('-w_init', '--w_init',  help="Weight initialization, default = uniform", default="uniform")
parser.add_argument('-cost_function', '--cost_function',  help="Cost function, default = squared_error", default="squared_error")
parser.add_argument('-seed', '--seed',  help="Seed for random number init., default = -1", default=-1)
parser.add_argument('-no_early_stop', '--no_early_stop', action="store_true", help="Turn off early stopping, default = False", default=False)
parser.add_argument('-continue_training', '--continue_training', action="store_true", help="Continue training, default = False", default=False)
args = parser.parse_args()



# get training data file:
if args.trainfile != None:
    trainfile = args.trainfile
    print("# Training data file: " + args.trainfile )
else:
    sys.stderr.write("Please specify training data file!\n")
    sys.exit(2)

# get validation data file:
if args.valfile != None:
    valfile = args.valfile
    print("# Validation data file: " + args.valfile )
else:
    sys.stderr.write("Please specify validation data file!\n")
    sys.exit(2)




try:
    MAX_PEP_SEQ_LEN=int(args.max_pep_seq_length)
    print("# max. peptide sequence length: " + str(MAX_PEP_SEQ_LEN))
except:
    sys.stderr.write("Problem with max. peptide sequence length specification (option -max_pep_seq_length)!\n")
    sys.exit(2)

try:
    MOTIF_LEN=int(args.motif_length)
    print("# mmotif length: " + str(MOTIF_LEN))
except:
    sys.stderr.write("Problem with motif length specification (option -motif_length)!\n")
    sys.exit(2)


try:
    BATCH_SIZE=int(args.batch_size)
    print("# batch size: " + str(BATCH_SIZE))
except:
    sys.stderr.write("Problem with mini batch size specification (option -batch_size)!\n")
    sys.exit(2)

try:
    EPOCHS=range(1, int(args.epochs)+1)
    print("# number of training epochs: " + str(args.epochs))
except:
    sys.stderr.write("Problem with epochs specification (option -epochs)!\n")
    sys.exit(2)

try:
    N_FILTERS=int(args.n_filters)
    print("# number of convolutional filters: " + str(N_FILTERS))
except:
    sys.stderr.write("Problem with number of CNN filters specification (option -n_filters)!\n")
    sys.exit(2)

try:
    LEARNING_RATE=float(args.learning_rate)
    print("# learning rate: " + str(LEARNING_RATE))
except:
    sys.stderr.write("Problem with learning rate specification (option -learning_rate)!\n")
    sys.exit(2)

try:
    UPDATE=args.update
    print("# weight update method: " + str(UPDATE))
except:
    sys.stderr.write("Problem with update specification (option -update)!\n")
    sys.exit(2)

try:
    ACTIVATION=args.activation
    print("# activation function: " + str(ACTIVATION))
except:
    sys.stderr.write("Problem with activation function specification (option -activation)!\n")
    sys.exit(2)

try:
    DROPOUT=float(args.dropout)
    print("# dropout: " + str(DROPOUT))
except:
    sys.stderr.write("Problem with dropout specification (option -dropout)!\n")
    sys.exit(2)

try:
    N_HID=int(args.n_hid)
    print("# number of hidden units: " + str(N_HID))
except:
    sys.stderr.write("Problem with number of hidden neurons specification (option -n_hid)!\n")
    sys.exit(2)


try:
    ARCHITECTURE=args.architecture
    print("# architecture: " + str(ARCHITECTURE))
except:
    sys.stderr.write("Problem with architecture specification (option -architecture)!\n")
    sys.exit(2)

try:
    W_INIT=args.w_init
    print("# w_init: " + str(W_INIT))
except:
    sys.stderr.write("Problem with weight initialization specification (option -w_init)!\n")
    sys.exit(2)

try:
    COST_FUNCTION=args.cost_function
    print("# cost_function: " + str(COST_FUNCTION))
except:
    sys.stderr.write("Problem with cost function specification (option -cost_function)!\n")
    sys.exit(2)

try:
    SEED=int(args.seed)
    print("# seed: " + str(SEED))
except:
    sys.stderr.write("Problem with seed specification (option -seed)!\n")
    sys.exit(2)

NO_EARLY_STOP = args.no_early_stop

if NO_EARLY_STOP == True:
    print("# Early stopping is turned OFF" )
else:
    print("# Early stopping is turned ON" )

CONTINUE = args.continue_training
if CONTINUE == True:
    print("# Training is CONTINUED" )
else:
    print("# New training (NOT continuing old training)" )


################################################################################
#	MAIN
################################################################################

if SEED != -1:
    print("# Setting seed for random number generation...")
    lasagne.random.set_rng(np.random.RandomState(seed=SEED))
    np.random.seed(seed=SEED) # for shuffling training examples

print("# Loading data...")

# read in data as a list of numpy ndarrays:
X_pep_train,X_mhc_train,y_train = data_io_func.netcdf2pep(trainfile)
X_pep_val,X_mhc_val,y_val = data_io_func.netcdf2pep(valfile)


# get MHC pseudo sequence length (assumes they all have the same length):
MHC_SEQ_LEN = X_mhc_train[0].shape[0]
# get target length:
T_LEN = y_train[0].shape[0]

N_SEQS_VAL = y_val[0].shape[0]

if MAX_PEP_SEQ_LEN == -1:
    # no length restraint -> find max length in dataset
    MAX_PEP_SEQ_LEN = max( len(max(X_pep_train, key=len)), len(max(X_pep_val, key=len)) )
else:
    # remove peptides with length longer than max peptide length:
    idx=[i for i,x in enumerate(X_pep_train) if len(x) > MAX_PEP_SEQ_LEN]

    X_pep_train = [i for j, i in enumerate(X_pep_train) if j not in idx]
    X_mhc_train = [i for j, i in enumerate(X_mhc_train) if j not in idx]
    y_train = [i for j, i in enumerate(y_train) if j not in idx]

    idx=[i for i,x in enumerate(X_pep_val) if len(x) > MAX_PEP_SEQ_LEN]

    X_pep_val = [i for j, i in enumerate(X_pep_val) if j not in idx]
    X_mhc_val = [i for j, i in enumerate(X_mhc_val) if j not in idx]
    y_val = [i for j, i in enumerate(y_val) if j not in idx]

# save sequences as np.ndarray instead of list of np.ndarrays:

X_pep_train = data_io_func.pad_seqs_T(X_pep_train, MAX_PEP_SEQ_LEN)
X_mhc_train = data_io_func.pad_seqs_T(X_mhc_train, MHC_SEQ_LEN)
X_pep_val = data_io_func.pad_seqs_T(X_pep_val, MAX_PEP_SEQ_LEN)
X_mhc_val = data_io_func.pad_seqs_T(X_mhc_val, MHC_SEQ_LEN)
N_FEATURES = X_pep_train.shape[1]

y_train = data_io_func.pad_seqs(y_train, T_LEN)
y_val = data_io_func.pad_seqs(y_val, T_LEN)


# Prepare Theano variables for targets and learning rate:
sym_target = T.vector('targets',dtype='float32')
sym_l_rate=T.scalar()


# Build the network:
print("# Building the network...")

if ARCHITECTURE == "cnn":
    network,in_pep,in_mhc = NN_func.build_cnn(n_features=N_FEATURES,
                            n_filters=N_FILTERS,
                            activation=ACTIVATION,
                            dropout=DROPOUT,
                            mhc_seq_len=MHC_SEQ_LEN,
                            n_hid=N_HID,
                            w_init=W_INIT)
    architecture=np.array([ARCHITECTURE])
    net_hyper_params=np.array([N_FEATURES, N_FILTERS, ACTIVATION, DROPOUT, N_HID, W_INIT])

else:
    sys.stderr.write("Unknown architecture specified (option -architecture)!\n")
    sys.exit(2)

print("# params: " + str(lasagne.layers.get_all_params(network)))
weights=list(lasagne.layers.get_all_params(network))
weights = [str(e) for e in weights ]
w=[i for i, j in enumerate(weights) if j == 'W' or j=='W_in_to_ingate' or j=='W_hid_to_ingate' or j=='W_in_to_forgetgate' or j=='W_hid_to_forgetgate' or j=='W_in_to_cell' or j=='W_hid_to_cell' or j=='W_in_to_outgate' or j=='W_hid_to_outgate' ]
print("# weights: " + str(w))
print("# number of layers: " + str(len(lasagne.layers.get_all_layers(network))) )
print("# number of parameters: " + str(lasagne.layers.count_params(network)))

# INITIALIZE VARIABLES ---------------------------------------------------------

if CONTINUE == True:
    # get current parameters:
    params = lasagne.layers.get_all_param_values(network)
    old_params = np.load(args.modelfile)['arr_0']

    # check if dimensions match:
    assert len(old_params) == len(params)
    for j in range(0,len(old_params)):
        assert old_params[j].shape == params[j].shape
    # set parameters in network:
    lasagne.layers.set_all_param_values(network, old_params)





print("# Compiling theano training and validation functions...")

# TRAINING FUNCTION -----------------------------------------------------------

prediction = lasagne.layers.get_output(network)

#loss function:
if COST_FUNCTION == "squared_error":
    loss = lasagne.objectives.squared_error(prediction.flatten(), sym_target)
else:
    sys.stderr.write("Unknown cost function specified (option -cost_function)!\n")
    sys.exit(2)
loss = loss.mean()


# update:
params = lasagne.layers.get_all_params(network, trainable=True)

if UPDATE == "sgd":
    updates = lasagne.updates.sgd(loss, params, learning_rate=sym_l_rate)
elif UPDATE == "rmsprop":
    updates = lasagne.updates.rmsprop(loss, params, learning_rate=sym_l_rate)
elif UPDATE == "adam":
    updates = lasagne.updates.adam(loss, params, learning_rate=sym_l_rate)
elif UPDATE == "adadelta":
    updates = lasagne.updates.adadelta(loss, params, learning_rate=sym_l_rate)
else:
    sys.stderr.write("Unknown update specified (option -update)!\n")
    sys.exit(2)

# compile training function
train_fn = theano.function([in_pep.input_var, in_mhc.input_var, sym_target, sym_l_rate], loss, updates=updates)


# VALIDATION FUNCTION ----------------------------------------------------------
test_prediction = lasagne.layers.get_output(network, deterministic=True)

if COST_FUNCTION == "squared_error":
    test_loss = lasagne.objectives.squared_error(test_prediction.flatten(),sym_target)
test_loss = test_loss.mean()


# compile validation function:
val_fn = theano.function([in_pep.input_var, in_mhc.input_var, sym_target], test_loss)

# TRAINING LOOP:----------------------------------------------------------------
start_time = time.time()


print("# Start training loop...")

if NO_EARLY_STOP == False:
    b_epoch=0
    b_train_err=99
    b_val_err=99

for e in EPOCHS:

    train_err = 0
    train_batches = 0
    val_err = 0
    val_batches = 0
    e_start_time = time.time()

    # shuffle training examples and iterate through minbatches:
    for batch in iterate_minibatches(X_pep_train, X_mhc_train, y_train, BATCH_SIZE):
        pep, mhc, targets = batch
        train_err += train_fn(pep, mhc, targets.flatten(), LEARNING_RATE)
        train_batches += 1

    # predict validation set:
    for batch in iterate_minibatches(X_pep_val, X_mhc_val, y_val, BATCH_SIZE):
        pep, mhc, targets = batch
        val_err += val_fn(pep, mhc, targets.flatten())
        val_batches += 1

    # save model:
    if NO_EARLY_STOP == True:
        np.savez(args.modelfile, lasagne.layers.get_all_param_values(network), architecture, net_hyper_params)
    else:
        # use early stopping, save only best model:
        if (val_err/val_batches) < b_val_err:
            np.savez(args.modelfile, lasagne.layers.get_all_param_values(network), architecture, net_hyper_params)
            b_val_err = val_err/val_batches
            b_train_err = train_err/train_batches
            b_epoch = e

    # print performance:
    print("Epoch " + str(e) +
    "\ttraining error: " + str(round(train_err/train_batches, 4)) +
    "\tvalidation error: " + str(round(val_err/val_batches, 4)) +
    "\ttime: " + str(round(time.time()-e_start_time, 3)) + " s")

    # calc stats:
    params = lasagne.layers.get_all_param_values(network)

# print best performance:
if NO_EARLY_STOP == False:
    print("# Best epoch: " + str(b_epoch) +
        "\ttrain error: " + str(round(b_train_err, 4)) +
        "\tvalidation error: " + str(round(b_val_err, 4)) )
# report total time used for training:
print("# Time for training: " + str(round((time.time()-start_time)/60, 3)) + " min" )


# UPDATE BATCH NORM STATS-------------------------------------------------------
print("# Updating batch norm stats...")


# load best parameters:
params = lasagne.layers.get_all_param_values(network)
best_params = np.load(args.modelfile)['arr_0']

# check if dimensions match:
assert len(best_params) == len(params)
for j in range(0,len(best_params)):
    assert best_params[j].shape == params[j].shape
# set parameters in network:
lasagne.layers.set_all_param_values(network, best_params)


# compile Batch norm update function:

prediction = lasagne.layers.get_output(network, deterministic=True, batch_norm_update_averages=True)

if COST_FUNCTION == "squared_error":
    loss = lasagne.objectives.squared_error(prediction.flatten(),sym_target)
loss = loss.mean()

bn_update_fn = theano.function([in_pep.input_var, in_mhc.input_var, sym_target], loss)


# predict training set:
for batch in iterate_minibatches(X_pep_train, X_mhc_train, y_train, len(X_pep_train)):
    pep, mhc, targets = batch
    tmp = bn_update_fn(pep, mhc, targets.flatten())

# save best parameters:
np.savez(args.modelfile, lasagne.layers.get_all_param_values(network), architecture, net_hyper_params)


print("# Done!")
