#!/usr/bin/env python

"""
Predict data with CNN trained using the Lasagne library:
https://github.com/Lasagne
"""

from __future__ import print_function
import argparse
import sys
import os
import time

import csv
import numpy as np
from scipy.io import netcdf
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import theano
import theano.tensor as T

import lasagne

import data_io_func
import NN_func



################################################################################
#	PARSE COMMANDLINE OPTIONS
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-data', '--datafile',  help="file with data to be predicted")
parser.add_argument('-data_aa', '--aafile',  help="file with data to be predicted")
parser.add_argument('-ensemblelist', '--ensemblelist',  help="text file containing list of hyper parameters and weight files")
parser.add_argument('-out', '--outfile',  help="file to store output table")
parser.add_argument('-max_pep_seq_length', '--max_pep_seq_length',  help="maximal peptide sequence length, default = -1", default=-1)
args = parser.parse_args()

# get data file:
if args.datafile != None:
    datafile = args.datafile
else:
    sys.stderr.write("Please specify data file!\n")
    sys.exit(2)

# get data file with AA sequences:
if args.aafile != None:
    aafile = args.aafile
else:
    sys.stderr.write("Please specify data file with AA sequences!\n")
    sys.exit(2)

# get ensemble list:
if args.ensemblelist != None:
    ensemblelist = args.ensemblelist
else:
    sys.stderr.write("Please specify data file with hyper parameters and weight files!\n")
    sys.exit(2)

# get outputfile:
if args.outfile != None:
    outfilename = args.outfile
else:
    sys.stderr.write("Please specify output file!\n")
    sys.exit(2)



try:
    MAX_PEP_SEQ_LEN=int(args.max_pep_seq_length)
except:
    sys.stderr.write("Problem with max. peptide sequence length specification (option -max_pep_seq_length)!\n")
    sys.exit(2)



################################################################################
#   READ ENSEMBLE FILE
################################################################################

# read list of ensembles:
ensembles=[]
with open(ensemblelist, 'rb') as infile:
    ensembles = list(csv.reader(infile, delimiter='\t'))
ensembles=filter(None,ensembles)


################################################################################
#   LOAD DATA
################################################################################

print("# Loading data...")

# read in data as a list of numpy ndarrays:
X_pep,X_mhc,y = data_io_func.netcdf2pep(datafile)


# get MHC pseudo sequence length (assumes they all have the same length):
MHC_SEQ_LEN = X_mhc[0].shape[0]
# get target length:
T_LEN = y[0].shape[0]


# find max peptide sequence length (if not specified)
if MAX_PEP_SEQ_LEN == -1:
    MAX_PEP_SEQ_LEN = len(max(X_pep, key=len))

# save sequences as np.ndarray instead of list of np.ndarrays:

X_pep_mp = data_io_func.pad_seqs_T(X_pep, MAX_PEP_SEQ_LEN)
X_mhc_mp = data_io_func.pad_seqs_T(X_mhc, MHC_SEQ_LEN)

y = data_io_func.pad_seqs(y, T_LEN)

# save Amino Acid seqeunces and MHC receptors:
pep_aa,mhc_molecule = data_io_func.get_pep_aa_mhc(aafile, MAX_PEP_SEQ_LEN)



################################################################################
#   PREDICT SINGLE NETWORKS
################################################################################

# variable to store predcitons:
all_pred = np.zeros(( len(ensembles),len(X_pep) ))

# go through each net and predict:
count=0
old_hyper_params=''

for l in ensembles:

    paramfile=l[0]

    # LOAD PARAMETERS:----------------------------------------------------------
    # load parameters of best model:
    best_params = np.load(paramfile)['arr_0']
    ARCHITECTURE = np.load(paramfile)['arr_1']
    hyper_params = np.load(paramfile)['arr_2']


    # BUILD NETWORK AND COMPILE TRAINING FUNCTION:------------------------------
    if set(hyper_params) != set(old_hyper_params):
        if ARCHITECTURE == "cnn":

            N_FEATURES=int(hyper_params[0])
            N_FILTERS=int(hyper_params[1])
            ACTIVATION=hyper_params[2]
            DROPOUT=float(hyper_params[3])
            N_HID=int(hyper_params[4])
            W_INIT=hyper_params[5]

            network,in_pep,in_mhc = NN_func.build_cnn(
                                    n_features=N_FEATURES,
                                    n_filters=N_FILTERS,
                                    activation=ACTIVATION,
                                    dropout=DROPOUT,
                                    mhc_seq_len=MHC_SEQ_LEN,
                                    n_hid=N_HID,
                                    w_init=W_INIT)

        else:
            sys.stderr.write("Unknown architecture specified (option -architecture)!\n")
            sys.exit(2)

        # COMPILE PREDICTION FUNCTION-----------------------------------------------

        prediction = lasagne.layers.get_output(network, deterministic=True)

        # compile validation function:
        pred_fn = theano.function([in_pep.input_var, in_mhc.input_var], prediction)

    # SET WEIGHTS---------------------------------------------------------------

    # get current parameters:
    params = lasagne.layers.get_all_param_values(network)

    # check if dimensions match:
    assert len(best_params) == len(params)
    for j in range(0,len(best_params)):
        assert best_params[j].shape == params[j].shape
    # set parameters in network:
    lasagne.layers.set_all_param_values(network, best_params)


    # RUN FORWARD PASS----------------------------------------------------------

    # predict validation set:
    if ARCHITECTURE == "cnn":
        all_pred[count] = pred_fn(X_pep_mp, X_mhc_mp).flatten()
    else:
        sys.stderr.write("Unknown data encoding in ensemble list!\n")
        sys.exit(2)

    old_hyper_params=hyper_params
    count +=1

# calculate mean predictions:
pred = np.mean(all_pred, axis=0)

################################################################################
#   PRINT RESULTS TABLE
################################################################################

print("# Printing results...")

assert pred.shape[0] == y.shape[0] == len(pep_aa) == len(mhc_molecule)
outfile = open(outfilename, "w")

outfile.write("peptide\tmhc\tprediction\ttarget\n")
y=y.flatten()
for i in range(0,len(pep_aa)):
    outfile.write(pep_aa[i] + "\t" + mhc_molecule[i] + "\t" + str(pred[i]) + "\t" + str(y[i]) + "\n")

# calculate PCC:
pcc,pval = pearsonr(pred.flatten(), y.flatten())
# calculate AUC:
y_binary = np.where(y>=0.42562, 1,0)
auc = roc_auc_score(y_binary.flatten(), pred.flatten())

outfile.write("# PCC: " + str(pcc) + " p-value: " + str(pval) + " AUC: " + str(auc) + "\n")
