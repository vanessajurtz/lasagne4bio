#!/usr/bin/env python

"""
Functions for data IO for neural network training.
"""

from __future__ import print_function
import argparse
import sys
import os
import time

from operator import add
import math
import numpy as np
from scipy.io import netcdf
import theano
import theano.tensor as T

import lasagne

theano.config.floatX='float32'

def netcdf2pep(filename):
    '''
    read peptide, MHC and target data from NetCDF file

    parameters:
        - filename : file in which data is stored
    returns:
        - peptides : list of np.ndarrays containing encoded peptide sequences
        - mhcs : list of np.ndarrays containing encoded MHC pseudo sequences
        - targets : list of np.ndarrays containing targets (log transformed IC 50 values)
    '''
    # open file:
    f = netcdf.netcdf_file(filename, 'r')

    # get peptide and MHC sequence lengths:
    tmp = f.variables['peplen']
    peplength = tmp[:].copy()

    tmp = f.variables['mhclen']
    mhclength = tmp[:].copy()

    p = 0
    m = 0
    peptides=[]
    mhcs=[]
    targets=[]

    for i in range(0, peplength.shape[0]):

        # get peptide seq as np.ndarray [AAs x encoding length]
        tmp = f.variables['peptide'].data[p:p + peplength[i]]
        peptides.append(tmp.astype(theano.config.floatX))
        p += peplength[i]

        # get MHC pseudo seq as np.ndarray [AAs x encoding length]
        tmp = f.variables['mhc'].data[m:m + mhclength[i]]
        mhcs.append(tmp.astype(theano.config.floatX))
        m += mhclength[i]

        # get target (one transformed IC 50 value per peptide)
        tmp = f.variables['target'].data[i]
        if len(tmp.shape) == 0:
            tmp=tmp.reshape(1,1)
        if len(tmp.shape) ==1:
            tmp.reshape(1,tmp.shape[0])
        targets.append(tmp.astype(theano.config.floatX))

    # close file:
    f.close()

    # return data:
    return peptides, mhcs, targets

# modified from nntools:--------------------------------------------------------
def pad_seqs(X, length):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - batch_size : int
            Mini-batch size

    returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
    '''

    n_seqs = len(X)
    n_features = X[0].shape[1]

    X_pad = np.zeros((n_seqs, length, n_features),
                       dtype=theano.config.floatX)
    for i in range(0,len(X)):
        X_pad[i, :X[i].shape[0], :n_features] = X[i]
    return X_pad

def pad_seqs_mask(X, length):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.

    returns:
        - X_pad : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Tensor denoting what to include,
            shape=(n_batches, batch_size, length, n_features)
    '''

    n_seqs = len(X)
    n_features = X[0].shape[1]

    X_pad = np.zeros((n_seqs, length, n_features),
                       dtype=theano.config.floatX)
    X_mask = np.zeros((n_seqs, length), dtype=np.bool)

    for i in range(0,len(X)):
        X_pad[i, :X[i].shape[0], :n_features] = X[i]
        X_mask[i, :X[i].shape[0]] = 1
    return X_pad,X_mask

def pad_pep_mhc_mask(X_pep, X_mhc, max_pep_seq_len, mhc_seq_len):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X_pep : list of np.ndarray
            List of matrices containing encoded peptide sequence
        - max_pep_seq_len : int
            Sequence length of peptides.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - X_mhc : list of np.ndarray
            List of matrices containing encoded MHC pseudo sequence
        - mhc_seq_len : int
            Sequence length of MHC.  Smaller sequences will be padded with 0s,
            longer will be truncated.

    returns:
        - X_pad : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Tensor denoting what to include,
            shape=(n_batches, batch_size, length, n_features)
    '''
    assert(len(X_pep) == len(X_mhc))
    n_seqs = len(X_pep)
    n_features = X_pep[0].shape[1]

    X_pad = np.zeros((n_seqs, mhc_seq_len + max_pep_seq_len +1, n_features),
                       dtype=theano.config.floatX)
    X_mask = np.zeros((n_seqs, mhc_seq_len + max_pep_seq_len +1), dtype=np.bool)

    for i in range(0,n_seqs):
        # MHC
        X_pad[i, :X_mhc[i].shape[0], :n_features] = X_mhc[i]
        X_mask[i, :X_mhc[i].shape[0]] = 1
        #space
        X_pad[i, X_mhc[i].shape[0]: (X_mhc[i].shape[0] + 1), :n_features] = 1
        X_mask[i, X_mhc[i].shape[0]:(X_mhc[i].shape[0] + 1)] = 1
        #peptide
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), :n_features] = X_pep[i]
        X_mask[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0])] = 1
    return X_pad,X_mask

def pad_pep_mhc_mask_final(X_pep, X_mhc, max_pep_seq_len, mhc_seq_len):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X_pep : list of np.ndarray
            List of matrices containing encoded peptide sequence
        - max_pep_seq_len : int
            Sequence length of peptides.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - X_mhc : list of np.ndarray
            List of matrices containing encoded MHC pseudo sequence
        - mhc_seq_len : int
            Sequence length of MHC.  Smaller sequences will be padded with 0s,
            longer will be truncated.

    returns:
        - X_pad : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Tensor denoting what to include,
            shape=(n_batches, batch_size, length, n_features)
            separate encoding of peptide and MHC sequence + peptide length encoding
    '''
    assert(len(X_pep) == len(X_mhc))
    n_seqs = len(X_pep)
    n_features = X_pep[0].shape[1]

    X_pad = np.zeros((n_seqs, mhc_seq_len + max_pep_seq_len +1, (2*n_features)),
                       dtype=theano.config.floatX)
    X_mask = np.zeros((n_seqs, mhc_seq_len + max_pep_seq_len +1), dtype=np.bool)

    for i in range(0,n_seqs):
        # MHC + mask:
        X_pad[i, :X_mhc[i].shape[0], :n_features] = X_mhc[i]
        X_mask[i, :X_mhc[i].shape[0]] = 1
        # blank in place for peptide encoding:
        X_pad[i, :X_mhc[i].shape[0], n_features:(2*n_features)] = 0

        #spacer between MHC and peptide:
        X_pad[i, X_mhc[i].shape[0]: (X_mhc[i].shape[0] + 1), :(2*n_features)] = 1
        X_mask[i, X_mhc[i].shape[0]:(X_mhc[i].shape[0] + 1)] = 1

        # peptide + mask:
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), n_features:(2*n_features)] = X_pep[i]
        X_mask[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0])] = 1
        # blank in place for MHC encoding:
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), :n_features] = 0

    return X_pad,X_mask

def pad_mhc_mask_final(X_mhc, mhc_seq_len):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X_mhc : list of np.ndarray
            List of matrices containing encoded MHC pseudo sequence
        - mhc_seq_len : int
            Sequence length of MHC.  Smaller sequences will be padded with 0s,
            longer will be truncated.

    returns:
        - X_pad : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Tensor denoting what to include,
            shape=(n_batches, batch_size, length, n_features)
            separate encoding of peptide and MHC sequence + peptide length encoding
    '''
    #assert(len(X_pep) == len(X_mhc))
    n_seqs = len(X_mhc)
    n_features = X_mhc[0].shape[1]

    X_pad = np.zeros((n_seqs, mhc_seq_len + 1, (2*n_features)),
                       dtype=theano.config.floatX)
    X_mask = np.zeros((n_seqs, mhc_seq_len + 1), dtype=np.bool)

    for i in range(0,n_seqs):
        # MHC + mask:
        X_pad[i, :X_mhc[i].shape[0], :n_features] = X_mhc[i]
        X_mask[i, :X_mhc[i].shape[0]] = 1
        # blank in place for peptide encoding:
        X_pad[i, :X_mhc[i].shape[0], n_features:(2*n_features)] = 0

        #spacer between MHC and peptide:
        X_pad[i, X_mhc[i].shape[0]: (X_mhc[i].shape[0] + 1), :(2*n_features)] = 1
        X_mask[i, X_mhc[i].shape[0]:(X_mhc[i].shape[0] + 1)] = 1

    return X_pad,X_mask


def pad_pep_mhc_mask_multi(X_pep, X_mhc, max_pep_seq_len, mhc_seq_len, n_aa):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X_pep : list of np.ndarray
            List of matrices containing encoded peptide sequence
        - max_pep_seq_len : int
            Sequence length of peptides.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - X_mhc : list of np.ndarray
            List of matrices containing encoded MHC pseudo sequence
        - mhc_seq_len : int
            Sequence length of MHC.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        -n_aa: number of AA to present in one time step.

    returns:
        - X_pad : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_seq, time_steps, n_features_new)
        - X_mask : np.ndarray
            Tensor denoting what to include,
            shape=(n_batches, batch_size, length, n_features)
            separate encoding of peptide and MHC sequence + peptide length encoding
    '''
    assert(len(X_pep) == len(X_mhc))
    n_seqs = len(X_pep)
    n_features = X_pep[0].shape[1]
    time_steps = int(math.ceil(mhc_seq_len / float(n_aa)) + math.ceil(max_pep_seq_len / float(n_aa)) + 1)
    ts_pep = int(math.ceil(mhc_seq_len / float(n_aa)))
    ts_mhc = int(math.ceil(max_pep_seq_len / float(n_aa)))

    X_pad = np.zeros((n_seqs, time_steps, (2*n_aa*n_features)),
                       dtype=theano.config.floatX)
    X_mask = np.zeros((n_seqs, time_steps), dtype=np.bool)

    for i in range(0,n_seqs):
        # MHC pseudo sequence:
        c=0
        for j in range(0, ts_mhc ):
            if c < X_mhc[i].shape[0]:
                tmp = X_mhc[i][c:min(c+n_aa,time_steps)].flatten()
                X_pad[i, j, :len(tmp)] = tmp
                X_mask[i, j] = 1

            c+=n_aa

        #spacer between MHC and peptide:
        X_pad[i, ts_mhc, :(2*n_aa*n_features)] = 1
        X_mask[i, ts_mhc] = 1

        # peptide sequence:
        c=0
        for j in range((ts_mhc +1) , time_steps):
            if c < X_pep[i].shape[0]:
                tmp = X_pep[i][c:min(c+n_aa,time_steps)].flatten()
                X_pad[i, j, (n_aa*n_features) : (n_aa*n_features+len(tmp))] = tmp
                X_mask[i, j] = 1

            c+=n_aa

    return X_pad,X_mask

def pad_pep_mhc_mask_sepw(X_pep, X_mhc, max_pep_seq_len, mhc_seq_len, motif_len):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X_pep : list of np.ndarray
            List of matrices containing encoded peptide sequence
        - max_pep_seq_len : int
            Sequence length of peptides.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - X_mhc : list of np.ndarray
            List of matrices containing encoded MHC pseudo sequence
        - mhc_seq_len : int
            Sequence length of MHC.  Smaller sequences will be padded with 0s,
            longer will be truncated.

    returns:
        - X_pad : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Tensor denoting what to include,
            shape=(n_batches, batch_size, length, n_features)
            separate encoding of peptide and MHC sequence + peptide length encoding
    '''
    assert(len(X_pep) == len(X_mhc))
    n_seqs = len(X_pep)
    n_features = X_pep[0].shape[1]

    X_pad = np.zeros((n_seqs, mhc_seq_len + max_pep_seq_len +1, (2*n_features+2)),
                       dtype=theano.config.floatX)
    X_mask = np.zeros((n_seqs, mhc_seq_len + max_pep_seq_len +1), dtype=np.bool)

    for i in range(0,n_seqs):
        # MHC + mask:
        X_pad[i, :X_mhc[i].shape[0], :n_features] = X_mhc[i]
        X_mask[i, :X_mhc[i].shape[0]] = 1
        # blank in place for peptide encoding:
        X_pad[i, :X_mhc[i].shape[0], n_features:(2*n_features)] = 0

        #spacer between MHC and peptide:
        X_pad[i, X_mhc[i].shape[0]: (X_mhc[i].shape[0] + 1), :(2*n_features)] = 1
        X_mask[i, X_mhc[i].shape[0]:(X_mhc[i].shape[0] + 1)] = 1

        # peptide + mask:
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), n_features:(2*n_features)] = X_pep[i]
        X_mask[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0])] = 1
        # blank in place for MHC encoding:
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), :n_features] = 0

        # peptide length encoding:
        o=(X_pep[i].shape[0]-motif_len)*1
        o=1/(1+math.exp(-o))
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), (2*n_features)] = o
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), (2*n_features+1)] = 1-o

    return X_pad,X_mask


def pad_pep_mhc_mask_sepw_pos(X_pep, X_mhc, max_pep_seq_len, mhc_seq_len, motif_len):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X_pep : list of np.ndarray
            List of matrices containing encoded peptide sequence
        - max_pep_seq_len : int
            Sequence length of peptides.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - X_mhc : list of np.ndarray
            List of matrices containing encoded MHC pseudo sequence
        - mhc_seq_len : int
            Sequence length of MHC.  Smaller sequences will be padded with 0s,
            longer will be truncated.

    returns:
        - X_pad : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Tensor denoting what to include,
            shape=(n_batches, batch_size, length, n_features)
            separate encoding of peptide and MHC sequence + peptide length encoding
    '''
    assert(len(X_pep) == len(X_mhc))
    n_seqs = len(X_pep)
    n_features = X_pep[0].shape[1]

    X_pad = np.zeros((n_seqs, mhc_seq_len + max_pep_seq_len +1, (2*n_features+4)),
                       dtype=theano.config.floatX)
    X_mask = np.zeros((n_seqs, mhc_seq_len + max_pep_seq_len +1), dtype=np.bool)

    for i in range(0,n_seqs):
        # MHC + mask:
        X_pad[i, :X_mhc[i].shape[0], :n_features] = X_mhc[i]
        X_mask[i, :X_mhc[i].shape[0]] = 1
        # blank in place for peptide encoding:
        X_pad[i, :X_mhc[i].shape[0], n_features:(2*n_features)] = 0

        #spacer between MHC and peptide:
        X_pad[i, X_mhc[i].shape[0]: (X_mhc[i].shape[0] + 1), :(2*n_features)] = 1
        X_mask[i, X_mhc[i].shape[0]:(X_mhc[i].shape[0] + 1)] = 1

        # peptide + mask:
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), n_features:(2*n_features)] = X_pep[i]
        X_mask[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0])] = 1
        # blank in place for MHC encoding:
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), :n_features] = 0

        # peptide length encoding:
        o=(X_pep[i].shape[0]-motif_len)*1
        o=1/(1+math.exp(-o))
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), (2*n_features)] = o
        X_pad[i, (X_mhc[i].shape[0] + 1):(X_mhc[i].shape[0] + 1 + X_pep[i].shape[0]), (2*n_features+1)] = 1-o

        for j in range(0, X_pep[i].shape[0]):
            X_pad[i, X_mhc[i].shape[0] + 1 + j, (2*n_features+2)] = float(j) / 15
            X_pad[i, X_mhc[i].shape[0] + 1 + j, (2*n_features+3)] = float(X_pep[i].shape[0]-j-1) / 15
        for j in range(0, X_mhc[i].shape[0]):
            X_pad[i, j, (2*n_features+2)] = float(j) / X_mhc[i].shape[0]
            X_pad[i, j, (2*n_features+3)] = float(X_mhc[i].shape[0]-j-1) / X_mhc[i].shape[0]

    return X_pad,X_mask



def pad_seqs_cnn_mask(X_pep, X_mhc, max_pep_seq_len, mhc_seq_len):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X_pep : list of np.ndarray
            List of matrices containing encoded peptide sequence
        - max_pep_seq_len : int
            Sequence length of peptides.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - X_mhc : list of np.ndarray
            List of matrices containing encoded MHC pseudo sequence
        - mhc_seq_len : int
            Sequence length of MHC.  Smaller sequences will be padded with 0s,
            longer will be truncated.

    returns:
        - X_pad_pep : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_seqs, n_features, length)
        - X_pad_mhc : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_seqs, n_features, length)
        - X_mask : np.ndarray
            Tensor denoting what to include after CNN before LSTM,
            shape=(n_seqs, length_cnn_out)
    '''
    assert(len(X_pep) == len(X_mhc))
    n_seqs = len(X_pep)
    n_features = X_pep[0].shape[1]

    X_pad_mhc = np.zeros((n_seqs, n_features, mhc_seq_len),
                       dtype=theano.config.floatX)
    X_pad_pep = np.zeros((n_seqs, n_features, max_pep_seq_len),
                       dtype=theano.config.floatX)
    mask_len = mhc_seq_len -3 +1 + \
                mhc_seq_len -8 +1 + \
                mhc_seq_len -9 +1 + \
                mhc_seq_len -10 +1 + \
                max_pep_seq_len -3 +1 + \
                max_pep_seq_len -8 +1 + \
                max_pep_seq_len -9 +1 + \
                max_pep_seq_len -10 +1 + \
                7*1

    X_mask = np.zeros((n_seqs, mask_len), dtype=np.bool)

    for i in range(0,n_seqs):
        # MHC + peptide:
        X_pad_mhc[i, :n_features, :X_mhc[i].shape[0]] = np.swapaxes(X_mhc[i],0,1)
        X_pad_pep[i, :n_features, :X_pep[i].shape[0]] = np.swapaxes(X_pep[i],0,1)

        # mask:
        start=0
        X_mask[i, start : X_mhc[i].shape[0] -3 +1 ] = 1
        start += (mhc_seq_len -3 +1)
        X_mask[i, start : start + 1 ] = 1
        start += 1
        X_mask[i, start : X_mhc[i].shape[0] -8 +1 ] = 1
        start += (mhc_seq_len -8 +1)
        X_mask[i, start : start + 1 ] = 1
        start += 1
        X_mask[i, start : X_mhc[i].shape[0] -9 +1 ] = 1
        start += (mhc_seq_len -9 +1)
        X_mask[i, start : start + 1 ] = 1
        start += 1
        X_mask[i, start : X_mhc[i].shape[0] -10 +1 ] = 1
        start += (mhc_seq_len -10 +1)
        X_mask[i, start : start + 1 ] = 1
        start += 1
        X_mask[i, start : X_pep[i].shape[0] -3 +1 ] = 1
        start += (max_pep_seq_len -3 +1)
        X_mask[i, start : start + 1 ] = 1
        start += 1
        X_mask[i, start : X_pep[i].shape[0] -8 +1 ] = 1
        start += (max_pep_seq_len -8 +1)
        X_mask[i, start : start + 1 ] = 1
        start += 1
        X_mask[i, start : X_pep[i].shape[0] -9 +1 ] = 1
        start += (max_pep_seq_len -9 +1)
        X_mask[i, start : start + 1 ] = 1
        start += 1
        X_mask[i, start : X_pep[i].shape[0] -10 +1 ] = 1

    return X_pad_pep,X_pad_mhc,X_mask


def pad_seqs_T(X, length):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - batch_size : int
            Mini-batch size

    returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, n_features, seqlength)
    '''

    n_seqs = len(X)
    n_features = X[0].shape[1]

    X_pad = np.zeros((n_seqs, n_features, length),
                       dtype=theano.config.floatX)
    for i in range(0,len(X)):
        slen=X[i].shape[0]
        X_pad[i, :n_features, :slen] = np.swapaxes(X[i],0,1)
    return X_pad

def pad_seqs_T_pl(X, length, motif_len):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - batch_size : int
            Mini-batch size

    returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, n_features, seqlength)
            with peptide sequence length encoding
    '''

    n_seqs = len(X)
    n_features = X[0].shape[1]

    X_pad = np.zeros((n_seqs, n_features, length),
                       dtype=theano.config.floatX)
    X_pl = np.zeros((n_seqs, 2),
                       dtype=theano.config.floatX)
    for i in range(0,len(X)):
        slen=X[i].shape[0]
        X_pad[i, :n_features, :slen] = np.swapaxes(X[i],0,1)

        # peptide length encoding:
        o=(slen-motif_len)*1
        o=1/(1+math.exp(-o))
        X_pl[i, 0] = o
        X_pl[i, 1] = 1-o

    return X_pad,X_pl

def pad_seqs_T_pl_pos(X, length, motif_len):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - batch_size : int
            Mini-batch size

    returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, n_features, seqlength)
            with peptide sequence length encoding
    '''

    n_seqs = len(X)
    n_features = X[0].shape[1]

    X_pad = np.zeros((n_seqs, n_features+4, length),
                       dtype=theano.config.floatX)

    for i in range(0,len(X)):
        # get peptide sequence length:
        slen=X[i].shape[0]

        # copy encoded sequence:
        X_pad[i, :n_features, :slen] = np.swapaxes(X[i],0,1)

        # peptide length encoding:
        o=(slen-motif_len)*1
        o=1/(1+math.exp(-o))

        X_pad[i, n_features, :slen] =  o
        X_pad[i, n_features+1, :slen] =  1-o

        # peptide position encoding:
        for j in range(0, slen):
            X_pad[i, n_features + 2, j] = float(j) / 15
            X_pad[i, n_features + 3, j] = float(slen-j-1) / 15


    return X_pad


def get_pep_aa_mhc(filename, MAX_PEP_SEQ_LEN):
    '''
    read AA seq of peptides and MHC molecule from text file

    parameters:
        - filename : file in which data is stored
    returns:
        - pep_aa : list of amino acid sequences of peptides (as string)
        - mhc_molecule : list of name of MHC molecules (string)
    '''
    pep_aa=[]
    mhc_molecule=[]
    infile = open(filename, "r")

    for l in infile:
        l=filter(None, l.strip().split())
        if len(l[0]) <= MAX_PEP_SEQ_LEN:
            pep_aa.append(l[0])
            mhc_molecule.append(l[2])
    infile.close()

    return pep_aa,mhc_molecule

def read_mhc_list(filename, mhc_allowed):
    '''
    read AA seq of MHC molecules from text file

    parameters:
        - filename : file in which data is stored
        - mhc_allowed : list of allowed MHC molecules
    returns:
        - X_mhc : list of amino acid sequences of MHCs (as string)
        - mhc_molecule : list of name of MHC molecules (string)
    '''
    X_mhc=[]
    mhc_molecule=[]
    infile = open(filename, "r")

    for l in infile:
        l=filter(None, l.strip().split())
        if l[0] in mhc_allowed:
            X_mhc.append(l[1])
            mhc_molecule.append(l[0])
    infile.close()

    return X_mhc,mhc_molecule

def encode_seq(Xin, max_pep_seq_len, blosum):
    '''
    encode AA seq of peptides using BLOSUM50

    parameters:
        - Xin : list of peptide sequences in AA
    returns:
        - Xout : encoded peptide seuqneces (batch_size, max_pep_seq_len, n_features)
    '''
    # read encoding matrix:
    n_features=len(blosum['A'])
    n_seqs=len(Xin)

    # make variable to store output:
    Xout = np.zeros((n_seqs, max_pep_seq_len, n_features),
                       dtype=theano.config.floatX)

    for i in range(0,len(Xin)):
        for j in range(0,len(Xin[i])):
            Xout[i, j, :n_features] = blosum[ Xin[i][j] ]
    return Xout


# feed forward:-----------------------------------------------------------------
def get_pep_mhc_target(filename, MAX_PEP_SEQ_LEN):
    '''
    read AA seq of peptides, MHC molecule and binding affinity from text file

    parameters:
        - filename : file in which data is stored
    returns:
        - pep_aa : list of amino acid sequences of peptides (list of strings)
        - mhc_molecule : list of name of MHC molecules (list of strings)
        - target: binding affinity (list of strings)
    '''
    pep_aa=[]
    mhc_molecule=[]
    targets=[]
    infile = open(filename, "r")

    for l in infile:
        l=filter(None, l.strip().split())
        if len(l[0]) <= MAX_PEP_SEQ_LEN:
            pep_aa.append(l[0])
            mhc_molecule.append(l[2])
            targets.append(l[1])
    infile.close()

    return pep_aa,mhc_molecule,targets

def read_blosum(filename):
    '''
    read in BLOSUM matrix

    parameters:
        - filename : file containing BLOSUM matrix

    returns:
        - blosum : dictionnary AA -> blosum encoding (as list)
    '''

    # read BLOSUM matrix:
    blosumfile = open(filename, "r")
    blosum = {}
    B_idx = []
    J_idx = []
    Z_idx = []
    star_idx = []

    for l in blosumfile:
        l = l.strip()

        if l[0] == '#':
            l = l.strip("#")
            l = l.split(" ")
            l = filter(None, l)
            if l[0] == "A":
                try:
                    B_idx = l.index('B')
                except:
                    B_idx = 99
                try:
                    J_idx = l.index('J')
                except:
                    J_idx = 99
                try:
                    Z_idx = l.index('Z')
                except:
                    Z_idx = 99
                star_idx = l.index('*')
        else:
            l = l.split(" ")
            l = filter(None, l)
            aa = str(l[0])
            if (aa != 'B') & (aa != 'J') & (aa != 'Z') & (aa != '*'):
                tmp = l[1:len(l)]
                # tmp = [float(i) for i in tmp]
                # get rid of BJZ*:
                tmp2 = []
                for i in range(0, len(tmp)):
                    if (i != B_idx) & (i != J_idx) & (i != Z_idx) & (i != star_idx):
                        tmp2.append(0.1*float(tmp[i])) # divide by 10

                #save in BLOSUM matrix
                blosum[aa]=tmp2
    blosumfile.close()
    return(blosum)


def read_blosum_np(filename):
    '''
    read in BLOSUM matrix

    parameters:
        - filename : file containing BLOSUM matrix

    returns:
        - blosum : numpy array containing BLOSUM matrix
    '''

    # read BLOSUM matrix:
    blosumfile = open(filename, "r")
    blosum = np.zeros((21, 21))
    B_idx = []
    J_idx = []
    Z_idx = []
    star_idx = []

    count=0

    for l in blosumfile:
        l = l.strip()

        if l[0] == '#':
            l = l.strip("#")
            l = l.split(" ")
            l = filter(None, l)
            if l[0] == "A":
                try:
                    B_idx = l.index('B')
                except:
                    B_idx = 99
                try:
                    J_idx = l.index('J')
                except:
                    J_idx = 99
                try:
                    Z_idx = l.index('Z')
                except:
                    Z_idx = 99
                star_idx = l.index('*')
        else:
            l = l.split(" ")
            l = filter(None, l)
            aa = str(l[0])
            if (aa != 'B') & (aa != 'J') & (aa != 'Z') & (aa != '*'):
                tmp = l[1:len(l)]
                # tmp = [float(i) for i in tmp]
                # get rid of BJZ*:
                tmp2 = []
                for i in range(0, len(tmp)):
                    if (i != B_idx) & (i != J_idx) & (i != Z_idx) & (i != star_idx):
                        tmp2.append(float(tmp[i]))

                #save in BLOSUM matrix
                blosum[count]=np.array(tmp2)
                count+=1
    blosumfile.close()
    return(blosum)



def read_blosum_MN(filename):
    '''
    read in BLOSUM matrix

    parameters:
        - filename : file containing BLOSUM matrix

    returns:
        - blosum : dictionnary AA -> blosum encoding (as list)
    '''

    # read BLOSUM matrix:
    blosumfile = open(filename, "r")
    blosum = {}
    B_idx = 99
    Z_idx = 99
    star_idx = 99

    for l in blosumfile:
        l = l.strip()

        if l[0] != '#':
            l= filter(None,l.strip().split(" "))

            if (l[0] == 'A') and (B_idx==99):
                B_idx = l.index('B')
                Z_idx = l.index('Z')
                star_idx = l.index('*')
            else:
                aa = str(l[0])
                if (aa != 'B') &  (aa != 'Z') & (aa != '*'):
                    tmp = l[1:len(l)]
                    # tmp = [float(i) for i in tmp]
                    # get rid of BJZ*:
                    tmp2 = []
                    for i in range(0, len(tmp)):
                        if (i != B_idx) &  (i != Z_idx) & (i != star_idx):
                            tmp2.append(float(tmp[i]))

                    #save in BLOSUM matrix
                    [i * 0.2 for i in tmp2] #scale (divide by 5)
                    blosum[aa]=tmp2
    blosumfile.close()
    return(blosum)



def read_real_blosum(filename):
    '''
    read in real value BLOSUM matrix

    parameters:
        - filename : file containing BLOSUM matrix

    returns:
        - real_blosum : dictionnary AA -> blosum encoding (as list)
    '''

    # read BLOSUM matrix:
    real_blosumfile = open(filename, "r")
    real_blosum = {}
    count=0
    AA=[]

    for l in real_blosumfile:
        l = l.strip()

        if l[0] != '#':
            l= filter(None,l.strip().split(" "))

            if l[0]=='A':
                AA=l
            else:
                l=[float(x) for x in l]
                real_blosum[AA[count]]=l
                count+=1
    real_blosumfile.close()
    return(real_blosum)


def read_MHC_pseudo_seq(filename):
    '''
    read in MHC pseudo sequence

    parameters:
        - filename : file containing MHC pseudo sequences

    returns:
        - mhc : dictionnary mhc -> AA sequence (as string)
        - mhc_seq_len : number of AA in mhc pseudo sequence
    '''
    # read MHC pseudo sequence:
    mhcfile=open(filename, "r")
    mhc={}
    mhc_seq_len=None
    for l in mhcfile:
        l=l.strip()
        l=l.split("\t")
        l=filter(None, l)

        mhc[l[0]]=l[1]
        if mhc_seq_len == None:
            mhc_seq_len = len(l[1])
    mhcfile.close()

    return mhc, mhc_seq_len


# def encode_PFR(pep,enc_mat,pfr_len,motif_len):
#     '''
#     encode PFR (peptide flanking regions)
#
#     parameters:
#         - peptide : list of strings (AA sequences of peptides)
#         - enc_mat: encoding matrix (real BLOSUM)
#         - length: length of PFR
#         - motif_len: length of binding core
#
#     returns:
#         - pfr : list of lists of lists: pfr[peptide][offset][left/right]
#     '''
#
#     pfr=[]
#     tmp_pfr=[]
#     enc_len=len(enc_mat[pep[0][0]])
#
#     for p in pep:
#         for i in range(0,len(p)-motif_len+1):
#             # initialize to 0:
#             l_pfr=[0] * enc_len
#             r_pfr=[0] * enc_len
#
#             # calculate left PFR:
#             for j in range(-pfr_len,0):
#                 if ((i+j) >=0) & ((i+j) < len(p)):
#                     l_pfr = map(add, l_pfr, enc_mat[p[i+j]])
#
#             # calculate right PFR:
#             for j in range(1,pfr_len+1):
#                 if ((i+motif_len+j) >=0) & ((i+motif_len+j) < len(p)):
#                     r_pfr = map(add, r_pfr, enc_mat[p[i+motif_len+j]])
#
#             # save:
#             tmp_pfr.append([l_pfr,r_pfr])
#         #save:
#         pfr.append(tmp_pfr)
#         tmp_pfr=[]
#     return pfr

def encode_PFR(pfr_seq,enc_mat):
    '''
    encode PFR (peptide flanking regions)

    parameters:
        - pfr_seq : list of strings (AA sequences of PFR)
        - enc_mat: encoding matrix (real BLOSUM)

    returns:
        - pfr : encoded PFR as list
    '''

    enc_len=len(enc_mat['A'])
    pfr=[0] * enc_len

    for i in pfr_seq:
        pfr = map(add, pfr, enc_mat[i])

    if len(pfr_seq)>0:
        pfr = [x * (1/float(len(pfr_seq))) for x in pfr] # divide by length of PFR sequence

    return pfr



def encode_mhc(mhc,mhc_mat,enc_mat):
    '''
    encode MHC input molecule

    parameters:
        - mhc: string (AA sequence of mhc molecule)
        - enc_mat: encoding matrix (BLOSUM)

    returns:
        - x : list containing encoded sequence
    '''
    x=[]
    mhc_seq=mhc_mat[mhc]
    for i in mhc_seq:
        x += enc_mat[i]
    return(x)

def encode_pep(pep,enc_mat,enc_mat_pfr,enc_pfr,pfr_len,offset,motif_len,pep_len,max_gap_len,max_ins_len):
    '''
    encode peptide sequence

    parameters:
        - pep: string (AA sequence of peptide)
        - enc_mat: encoding matrix (BLOSUM)
        - enc_mat_pfr: encoding matrix for PFR (real BLOSUM)
        - enc_pfr: True/False encode PFR?
        - pfr_len: max length of encoded PFR
        - offset: start position of motif within whole peptide
        - motif_len: length of binding core
        - pep_len: length of whole peptide
        - max_ins_len: maximal insert length
        - max_gap_len: maximal gap length

    returns:
        - x : list containing encoded sequence
    '''
    x=[]

    # left PFR:
    if (enc_pfr==True):
        pfr_seq = pep[max(0,offset-pfr_len):offset]
        x += encode_PFR(pfr_seq,enc_mat_pfr)
        #x += pfr[offset][0]
    # peptide:
    for i in pep:
        x += enc_mat[i]
    # right PFR:
    if (enc_pfr==True):
        if (offset + motif_len) < pep_len:
            pfr_seq = pep[(offset + motif_len) : max(pep_len,offset + motif_len + pfr_len)]
        else:
            pfr_seq=''
        x += encode_PFR(pfr_seq,enc_mat_pfr)
        #x += pfr[offset][1]

    # PFR length encoding:
    if (enc_pfr==True):
        ll=offset
        if ll > pfr_len:
            ll=pfr_len
        elif ll < 0:
            ll=0

        lr = pep_len - offset - motif_len
        if lr > pfr_len:
            lr=pfr_len
        elif lr <0:
            lr=0

        o = (pfr_len -ll)*1.0/pfr_len
        x += [o,(1-o)]

        o = (pfr_len -lr)*1.0/pfr_len
        x += [o,(1-o)]
    else:
        ll=max(0,offset)
        lr=max(0,(pep_len - offset -motif_len))

        o = (ll*1.0)/(ll+1)
        x += [o,(1-o)]

        o = (lr*1.0)/(lr+1)
        x += [o,(1-o)]

    # peptide length encoding (make this optional!):
    o = (pep_len - motif_len)*1.0
    o = 1/(1+math.exp(-o))
    x += [o,(1-o)]

    # gap length encoding:
    #if(max_gap_len >0):
    x += [0,1]
    # insertion length encoding:
    #if(max_ins_len >0):
    x += [0,1]
    # gap position encoding (make optional!!)
    #if(max_gap_len >0):
    x += [1,0]

    # return encoded peptide:
    return(x)

def encode_pep_new(pep,enc_mat,enc_mat_pfr,enc_pfr,pfr_len,offset,motif_len,pep_len,max_gap_len,max_ins_len):
    '''
    encode peptide sequence

    parameters:
        - pep: string (AA sequence of peptide)
        - enc_mat: encoding matrix (BLOSUM)
        - enc_mat_pfr: encoding matrix for PFR (real BLOSUM)
        - enc_pfr: True/False encode PFR?
        - pfr_len: max length of encoded PFR
        - offset: start position of motif within whole peptide
        - motif_len: length of binding core
        - pep_len: length of whole peptide
        - max_ins_len: maximal insert length
        - max_gap_len: maximal gap length

    returns:
        - x : list containing encoded sequence
    '''
    x=[]

    # peptide:
    for i in pep:
        x += enc_mat[i]

    # peptide length encoding (make this optional!):
    o = (pep_len - motif_len)*1.0
    o = 1/(1+math.exp(-o))
    x += [o,(1-o)]

    # peptide position encoding:
    b = float(offset)/15
    e = float(pep_len - offset -1) /15
    x += [b,e]

    # return encoded peptide:
    return(x)



def encode_insertion(pep,enc_mat,enc_mat_pfr,enc_pfr,pfr_len,offset,motif_len,pep_len,max_gap_len,max_ins_len,ip,il):
    '''
    encode peptide sequence

    parameters:
        - pep: string (AA sequence of peptide)
        - enc_mat: encoding matrix (BLOSUM)
        - enc_mat_pfr: encoding matrix for PFR (real BLOSUM)
        - enc_pfr: True/False encode PFR?
        - pfr_len: max length of encoded PFR
        - offset: start position of motif within whole peptide
        - motif_len: length of binding core
        - pep_len: length of whole peptide
        - max_ins_len: maximal insert length
        - max_gap_len: maximal gap length
        - ip: insertion position
        - il: insertion length

    returns:
        - x : list containing encoded sequence
    '''
    x=[]

    # left PFR:
    if (enc_pfr==True):
        pfr_seq = pep[max(0,offset-pfr_len):offset]
        x += encode_PFR(pfr_seq,enc_mat_pfr)
    # peptide:
    for i in range(0,len(pep)):
        if i==ip:
            x += enc_mat['X'] * il # insertion
        x += enc_mat[pep[i]]
    # right PFR:
    if (enc_pfr==True):
        if (offset + motif_len) < pep_len:
            pfr_seq = pep[(offset + motif_len-il) : max(pep_len,offset + motif_len -il + pfr_len)]
        else:
            pfr_seq=''
        x += encode_PFR(pfr_seq,enc_mat_pfr)

    # PFR length encoding:
    if (enc_pfr==True):
        ll=offset
        if ll > pfr_len:
            ll=pfr_len
        elif ll < 0:
            ll=0

        lr = pep_len - offset - motif_len + il
        if lr > pfr_len:
            lr=pfr_len
        elif lr <0:
            lr=0

        o = (pfr_len -ll)*1.0/pfr_len
        x += [o,(1-o)]

        o = (pfr_len -lr)*1.0/pfr_len
        x += [o,(1-o)]
    else:
        ll=max(0,offset)
        lr=max(0,(pep_len - offset -motif_len + il))

        o = (ll*1.0)/(ll+1)
        x += [o,(1-o)]

        o = (lr*1.0)/(lr+1)
        x += [o,(1-o)]

    # peptide length encoding (make this optional!):
    o = (pep_len - motif_len)*1.0
    o = 1/(1+math.exp(-o))
    x += [o,(1-o)]

    # gap length encoding:
    #if(max_gap_len >0):
    x += [0,1]
    # insertion length encoding:
    #if(max_ins_len >0):
    o = (il*1.0)/max_ins_len
    x += [o,1-o]
    # insert/gap position encoding (make optional!!)
    #if(max_gap_len >0):
    o = (motif_len-1-ip)*1.0/(motif_len-1)
    x += [o,1-o]

    # return encoded peptide:
    return(x)


def encode_gap(pep,enc_mat,enc_mat_pfr,enc_pfr,pfr_len,offset,motif_len,pep_len,max_gap_len,max_ins_len,gp,gl):
    '''
    encode peptide sequence

    parameters:
        - pep: string (AA sequence of peptide)
        - enc_mat: encoding matrix (BLOSUM)
        - enc_mat_pfr: encoding matrix for PFR (real BLOSUM)
        - enc_pfr: True/False encode PFR?
        - pfr_len: max length of encoded PFR
        - offset: start position of motif within whole peptide
        - motif_len: length of binding core
        - pep_len: length of whole peptide
        - max_ins_len: maximal insert length
        - max_gap_len: maximal gap length
        - gp: gap position
        - gl: gap length

    returns:
        - x : list containing encoded sequence
    '''
    x=[]

    # left PFR:
    if (enc_pfr==True):
        pfr_seq = pep[max(0,offset-pfr_len):offset]
        x += encode_PFR(pfr_seq,enc_mat_pfr)
    # peptide:
    for i in range(0,len(pep)):
        if (i >= gp) & (i < gp + gl):
            next
        else:
            x += enc_mat[pep[i]]
    # right PFR:
    if (enc_pfr==True):
        if (offset + motif_len) < pep_len:
            pfr_seq = pep[(offset + motif_len + gl) : max(pep_len,offset + motif_len + gl + pfr_len)]
        else:
            pfr_seq=''
        x += encode_PFR(pfr_seq,enc_mat_pfr)

    # PFR length encoding:
    if (enc_pfr==True):
        ll=offset
        if ll > pfr_len:
            ll=pfr_len
        elif ll < 0:
            ll=0

        lr = pep_len - offset - motif_len - gl
        if lr > pfr_len:
            lr=pfr_len
        elif lr <0:
            lr=0

        o = (pfr_len -ll)*1.0/pfr_len
        x += [o,(1-o)]

        o = (pfr_len -lr)*1.0/pfr_len
        x += [o,(1-o)]
    else:
        ll=max(0,offset)
        lr=max(0,(pep_len - offset -motif_len - gl))

        o = (ll*1.0)/(ll+1)
        x += [o,(1-o)]

        o = (lr*1.0)/(lr+1)
        x += [o,(1-o)]

    # peptide length encoding (make this optional!):
    o = (pep_len - motif_len)*1.0
    o = 1/(1+math.exp(-o))
    x += [o,(1-o)]

    # gap length encoding:
    #if(max_gap_len >0):
    o = (min( 2,gl)*1.0)/min(2,max_gap_len) # mortens way
        #o = (gl*1.0)/ float(max_gap_len) #
    x += [o,1-o]
    # insertion length encoding:
    #if(max_ins_len >0):
    x += [0,1]
    # gap position encoding (make optional!!)
    #if(max_gap_len >0):
    o = (motif_len-1-gp)*1.0/(motif_len-1)
    x += [o,1-o]

    # return encoded peptide:
    return(x)


def encode_input(pep,mhc,enc_mat,enc_mat_pfr,mhc_mat,enc_mhc,enc_pfr,pfr_len,max_gap_len, max_ins_len,motif_len):
    '''
    encode peptide + MHC input to neural network

    parameters:
        - pep: string (AA sequence of peptide)
        - mhc: string (AA sequence of mhc molecule)
        - enc_mhc: True/False encode MHC?
        #- pfr: PFR (peptide flanking region)
        - enc_pfr: True/False encode PFR?
        - enc_mat: encoding matrix (BLOSUM)
        - enc_mat_pfr: encoding matrix (real BLOSUM)
        - pfr_len: max length of PFR
        - length: length of PFR
        - motif_len: length of binding core

    returns:
        - X : np.array input to neural network list of lists of lists: pfr[peptide][offset][left/right]
    '''
    X=[]

    # pre-encode MHC sequence:
    x_mhc=[]
    if enc_mhc==True:
        x_mhc=encode_mhc(mhc=mhc,mhc_mat=mhc_mat,enc_mat=enc_mat)

    if len(pep) < motif_len:
        # encode inserion at each position:
        for i in range(0,len(pep)):
            x = encode_insertion(pep=pep,enc_mat=enc_mat,enc_mat_pfr=enc_mat_pfr,enc_pfr=enc_pfr,
                            pfr_len=pfr_len,offset=0,motif_len=motif_len,
                            pep_len=len(pep),max_gap_len=max_gap_len,
                            max_ins_len=max_ins_len,ip=i,il=motif_len-len(pep))
            if enc_mhc==True:
                x += x_mhc
            X.append(x)
    else:
        # ungapped
        for o in range(0,len(pep)-motif_len+1):
            p=pep[o:(o+motif_len)]
            x = encode_pep(pep=p,enc_mat=enc_mat,enc_mat_pfr=enc_mat_pfr,enc_pfr=enc_pfr,
                            pfr_len=pfr_len,offset=o,motif_len=motif_len,
                            pep_len=len(pep),max_gap_len=max_gap_len,
                            max_ins_len=max_ins_len)

            if enc_mhc==True:
                x += x_mhc

            X.append(x)
        # gapped - deletions
        for gl in range(1,max_gap_len+1):
            for o in range(0,len(pep)-motif_len - gl +1):
                p=pep[o:(o+motif_len+gl)]
                for i in range(1,motif_len):
                    x = encode_gap(pep=p,enc_mat=enc_mat,enc_mat_pfr=enc_mat_pfr,enc_pfr=enc_pfr,
                                    pfr_len=pfr_len,offset=o,motif_len=motif_len,
                                    pep_len=len(pep),max_gap_len=max_gap_len,
                                    max_ins_len=max_ins_len,gp=i,gl=gl)

                    if enc_mhc==True:
                        x += x_mhc

                    X.append(x)
        # insertions
        for il in range(1,max_ins_len+1):
            for o in range(0,len(pep)-motif_len+il+1):
                p=pep[o:(o+motif_len-il)]
                for i in range(1,motif_len-il):
                    x = encode_insertion(pep=p,enc_mat=enc_mat,enc_mat_pfr=enc_mat_pfr,enc_pfr=enc_pfr,
                                    pfr_len=pfr_len,offset=o,motif_len=motif_len,
                                    pep_len=len(pep),max_gap_len=max_gap_len,
                                    max_ins_len=max_ins_len,ip=i,il=il)

                    if enc_mhc==True:
                        x += x_mhc

                    X.append(x)
    # convert to numpy array:
    X=np.array(X)
    return X


def encode_input_mhc_pos(pep,mhc,enc_mat,enc_mat_pfr,mhc_mat,enc_mhc,enc_pfr,pfr_len,max_gap_len, max_ins_len,motif_len):
    '''
    encode peptide + MHC input to neural network

    parameters:
        - pep: string (AA sequence of peptide)
        - mhc: string (AA sequence of mhc molecule)
        - enc_mhc: True/False encode MHC?
        #- pfr: PFR (peptide flanking region)
        - enc_pfr: True/False encode PFR?
        - enc_mat: encoding matrix (BLOSUM)
        - enc_mat_pfr: encoding matrix (real BLOSUM)
        - pfr_len: max length of PFR
        - length: length of PFR
        - motif_len: length of binding core

    returns:
        - X : np.array input to neural network list of lists of lists: pfr[peptide][offset][left/right]
    '''
    X=[]

    # pre-encode MHC sequence:
    x_mhc=[]
    if enc_mhc==True:
        x_mhc=encode_mhc(mhc=mhc,mhc_mat=mhc_mat,enc_mat=enc_mat)


    # ungapped
    for o in range(0,len(pep)-motif_len+1):
        p=pep[o:(o+motif_len)]
        x = encode_pep_new(pep=p,enc_mat=enc_mat,enc_mat_pfr=enc_mat_pfr,enc_pfr=enc_pfr,
                        pfr_len=pfr_len,offset=o,motif_len=motif_len,
                        pep_len=len(pep),max_gap_len=max_gap_len,
                        max_ins_len=max_ins_len)

        if enc_mhc==True:
            x += x_mhc

        X.append(x)

    # convert to numpy array:
    X=np.array(X)
    return X
