#!/usr/bin/env python

"""
Combine ensembles based on pred files
"""

from __future__ import print_function
import argparse
import sys
import os
import time
import gc

import csv
import numpy as np
from scipy.io import netcdf
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


################################################################################
#	PARSE COMMANDLINE OPTIONS
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-infile', '--infile',  help="file with list of files and weigts")
parser.add_argument('-out', '--outfile',  help="file to store output table")
args = parser.parse_args()

# get data file:
if args.infile != None:
    infilename = args.infile
else:
    sys.stderr.write("Please specify inputfile!\n")
    sys.exit(2)

# get outputfile:
if args.outfile != None:
    outfilename = args.outfile
else:
    sys.stderr.write("Please specify output file!\n")
    sys.exit(2)

################################################################################
#   READ ENSEMBLE FILE
################################################################################

# read list of ensembles:
files=[]
with open(infilename, 'rb') as infile:
    files = list(csv.reader(infile, delimiter='\t'))
files=filter(None,files)

infile.close()

################################################################################
#   READ DATA
################################################################################

# read first file:
infile=open(files[0][0],'r')

pep=[]
mhc=[]
pred=[]
target=[]

for l in infile:
    if l[0] != '#':
        l=filter(None,l.strip().split('\t'))
        if l[0]!= "peptide":
            pep.append(l[0])
            mhc.append(l[1])
            pred.append(float(l[2]))
            target.append(float(l[3]))
infile.close()

# get weight:
weights=[]
weights.append(float(files[0][1]))

# remove processed example
del files[0]

# create variable to store all predictions:
all_pred = np.zeros(( len(files)+1,len(pred) ))
all_pred[0]=pred

# process rest of predictions:
count=1
for f in files:
    # get weight:
    weights.append(float(f[1]))

    # get predictions:
    infile=open(f[0],'r')
    idx=0
    for l in infile:
        if l[0] != '#':
            l=filter(None,l.strip().split('\t'))
            if l[0]!= "peptide":
                # print(idx)
                # print(count)
                # print(l)
                # print(all_pred.shape)
                all_pred[count,idx]=float(l[2])
                idx+=1
    infile.close()

    count+=1

# scale predictions with weights and calulate mean:
weights=np.array(weights)
pred= np.divide(np.transpose(all_pred).dot(weights),np.sum(weights))

################################################################################
#   PRINT RESULTS TABLE
################################################################################

print("# Printing results...")

assert pred.shape[0] == len(target) == len(pep) == len(mhc)
outfile = open(outfilename, "w")

outfile.write("peptide\tmhc\tprediction\ttarget\n")
target=[float(i) for i in target]
target=np.array(target)
for i in range(0,len(pep)):
    outfile.write(pep[i] + "\t" + mhc[i] + "\t" + str(pred[i]) + "\t" + str(target[i]) + "\n")

# calculate PCC:
pcc,pval = pearsonr(pred.flatten(), target.flatten())
# calculate AUC:
target_binary = np.where(target>=0.42562, 1,0)
auc = roc_auc_score(target_binary.flatten(), pred.flatten())

outfile.write("# PCC: " + str(pcc) + " p-value: " + str(pval) + " AUC: " + str(auc) + "\n")
