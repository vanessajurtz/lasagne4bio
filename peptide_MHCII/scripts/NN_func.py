#!/usr/bin/env python

"""
Functions for specifiying convolutional neural network architectures.
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

theano.config.floatX='float32'

def set_activation(activation):
    '''
    Set activation function

    parameters:
        - activation : string (name of activation function)
    returns:
        - lasagne option
    '''
    # set activation function:
    if activation == "sigmoid":
        return lasagne.nonlinearities.sigmoid
    elif activation == "rectify":
        return lasagne.nonlinearities.rectify
    elif activation == "leaky_rectify":
        return lasagne.nonlinearities.leaky_rectify
    elif activation == "very_leaky_rectify":
        return lasagne.nonlinearities.very_leaky_rectify
    elif activation == "tanh":
        return lasagne.nonlinearities.tanh
    else:
        sys.stderr.write("Unknown activation function (option -activation)!\n")
        sys.exit(2)

def set_w_init(w_init):
    '''
    Set weight initilization

    parameters:
        - w_init : string (name of weight initilization)
    returns:
        - lasagne option
    '''
    if w_init == "normal":
        return lasagne.init.Normal(std=0.05)
    elif w_init == "uniform":
        return lasagne.init.Uniform(range=0.05)
    elif w_init == "constant":
        return lasagne.init.Constant()
    elif w_init == "glorot_normal":
        return lasagne.init.GlorotNormal()
    elif w_init == "glorot_uniform":
        return lasagne.init.GlorotUniform()
    elif w_init == "choice":
        return lasagne.init.Choice()
    else:
        sys.stderr.write("Unknown weight initilization (option -w_init)!\n")
        sys.exit(2)

def build_cnn( n_features, n_filters, activation, dropout, mhc_seq_len,
                n_hid, w_init):
    # set activation function:
    nonlin = set_activation(activation)

    # set weight initilization:
    winit = set_w_init(w_init)

    # CNN ---------------------------------------------------------------------
    # input layer:
    l_in_pep = lasagne.layers.InputLayer(
            shape=(None, n_features, None))

    # convolutional layer with motif length AA filters:
    l_conv_pep_9 = lasagne.layers.Conv1DLayer(
            l_in_pep,
            num_filters= n_filters,
            filter_size=9,
            stride=1,
            pad='valid',
            nonlinearity=nonlin,
            W=winit)
    # add a max pool layer here, outdim: (batch_size, n_filters):
    l_conv_pep_9_mp = lasagne.layers.GlobalPoolLayer(
            l_conv_pep_9,
            pool_function=theano.tensor.max)

    # input layer for MHC pseudo sequence:
    l_in_mhc = lasagne.layers.InputLayer(
            shape=(None, n_features, mhc_seq_len))
    l_in_mhc_flat = lasagne.layers.FlattenLayer(l_in_mhc) #(batch, mhc_seq_len * n_features)

    # DENSE -------------------------------------------------------------------
    # elementwise sum instead of concat:
    # hidden layer:
    l_dense_conv_pep_9 = lasagne.layers.DenseLayer(
             l_conv_pep_9_mp,
             num_units=n_hid,
             nonlinearity=lasagne.nonlinearities.linear,
             W=winit)
    l_dense_mhc = lasagne.layers.DenseLayer(
             l_in_mhc_flat,
             num_units=n_hid,
             nonlinearity=lasagne.nonlinearities.linear,
             W=winit)
    l_dense_all = lasagne.layers.ElemwiseSumLayer(
            incomings=(l_dense_mhc,
                        l_dense_conv_pep_9))
    l_dense_all_bn = lasagne.layers.batch_norm(l_dense_all)
    l_dense = lasagne.layers.NonlinearityLayer(
            incoming= l_dense_all_bn,
            nonlinearity=nonlin)
    # batch normalization:
    l_dense_bn = lasagne.layers.batch_norm(l_dense)
    # add a dropout layer:
    l_dense_drop = lasagne.layers.DropoutLayer(l_dense_bn, p=dropout)

    # output layer:
    l_out = lasagne.layers.DenseLayer(
            l_dense_drop,
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=winit)

    return l_out,l_in_pep,l_in_mhc


# define LSTM network:-----------------------------------------------------
def build_lstm( n_features, n_lstm, activation, dropout, n_hid, w_init):
    # set activation function:
    nonlin = set_activation(activation)

    # set weight initilization:
    winit = set_w_init(w_init)

    # input layer:
    l_in = lasagne.layers.InputLayer((None,None,n_features))
    # mask layer:
    l_in_mask = lasagne.layers.InputLayer((None,None))

    # LSTM layer:
    l_lstm = lasagne.layers.LSTMLayer(
            l_in,
            num_units=n_lstm,
            peepholes=False,
            learn_init=True,
            nonlinearity=lasagne.nonlinearities.tanh,
            mask_input=l_in_mask)
    # slice layer:
    l_slice = lasagne.layers.SliceLayer(
            l_lstm,
            axis=1,
            indices=-1)

    # batch normalization:
    l_slice_bn = lasagne.layers.batch_norm(l_slice)

    # hidden layer:
    l_dense = lasagne.layers.DenseLayer(
             l_slice_bn,
             num_units=n_hid,
             nonlinearity=nonlin,
             W=winit)
    # batch normalization:
    l_dense_bn = lasagne.layers.batch_norm(l_dense)
    # add a dropout layer:
    l_dense_drop = lasagne.layers.DropoutLayer(l_dense_bn, p=dropout)
    # output layer:
    l_out = lasagne.layers.DenseLayer(
            l_dense_drop,
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=winit)
    return l_out,l_in,l_in_mask
