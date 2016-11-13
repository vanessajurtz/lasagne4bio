# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:51:13 2015

@author: s145706
"""
import sys
import numpy as np
import string

if not (len(sys.argv) == 3):
    sys.exit("Usage: python debug_metadata.py <topX> <metadata_path>")

topX = int(sys.argv[1])
metadata_path = sys.argv[2]

print "Loading metadata file %s" % metadata_path
metadata = np.load(metadata_path)
acc_eval_valid = metadata['accuracy_eval_valid']
acc_train = metadata['accuracy_train']

print "Valid acc"
i = 1
for valid, train in zip(acc_eval_valid, acc_train):
    print "%d: %.5f %.5f" %(i, train, valid)
    i = i+1

# Running through top X best and making an averaged top X validations
best_val = np.zeros(shape=topX, dtype='float32')
max_val = np.zeros(shape=topX, dtype=int)
print
print "Max valids found at"
for i in range(topX):
    max_val[i] = np.argmax(acc_eval_valid)
    best_val[i] = acc_eval_valid[max_val[i]]
    acc_eval_valid[max_val[i]] = 0.0 # To make it take the next max value
    print max_val[i]

print "Best"
print "Validation = %.5f" % best_val[0]
print "Averaged on top %d" % topX
print "Validation = %.5f" % best_val.mean()
