#!/usr/bin/env python

from __future__ import print_function
import argparse
import subprocess
import sys
import os
import numpy as np
from scipy.io import netcdf

##########################################################################
#   FUNCTIONS
##########################################################################


def read_blosum(filename):
    '''
    read in BLOSUM matrix

    parameters:
        - filename : file containing BLOSUM matrix

    returns:
        - blosum : dictionnary AA -> blosum encoding (as array)
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
        tmp=l
        l=tmp.split(" ")
        if len(l) < 2:
            l=tmp.split("\t")
        l=filter(None, l)

        mhc[l[0]]=l[1]
        if mhc_seq_len == None:
            mhc_seq_len = len(l[1])
    mhcfile.close()

    return mhc, mhc_seq_len


def pep2netcdf(filename, peptides, peplength, mhcseqs, mhclength, targets):
    '''
    save peptide, MHC and target data as NetCDF file

    parameters:
        - filename : file to store data in
        - peptides : np.ndarray containing encoded peptide sequences
        - peplength : np.ndarray containing length of each peptide
        - mhcseqs : np.ndarray containing encoded MHC pseudo sequences
        - mhclength : np.ndarray containing length of each MHC sequence
        - targets : np.ndarray containing targets (log transformed IC 50 values)
    '''
    # open file:
    f=netcdf.netcdf_file(filename, 'w')

    # save targets:
    f.createDimension('target', len(targets))
    target=f.createVariable('target', 'f', ('target',))
    target[:]=targets

    # save sequence lengths:
    f.createDimension('peplen', len(peplength))
    peplen=f.createVariable('peplen', 'i', ('peplen',))
    peplen[:]=peplength
    # save peptides:
    f.createDimension('n', peptides.shape[0])
    f.createDimension('s', peptides.shape[1])
    peptide=f.createVariable('peptide', 'f', ('n', 's'))
    peptide[:][:]=peptides

    # save sequence lengths:
    f.createDimension('mhclen', len(mhclength))
    mhclen=f.createVariable('mhclen', 'i', ('mhclen',))
    mhclen[:]=mhclength
    # save peptides:
    f.createDimension('m', mhcseqs.shape[0])
    f.createDimension('l', mhcseqs.shape[1])
    mhc=f.createVariable('mhc', 'f', ('m', 'l'))
    mhc[:][:]=mhcseqs

    # close file:
    f.close()

##########################################################################
#       PARSE COMMANDLINE OPTIONS
##########################################################################

parser=argparse.ArgumentParser()
parser.add_argument('-i', '--infile',  help="AA peptide + affinity data")
parser.add_argument('-o', '--outfile', help="outputfile")
parser.add_argument('-m', '--mhc', help="MHC pseudo sequences")
parser.add_argument('-b', '--blosum', help="file with BLOSUM matrix")
args=parser.parse_args()

# open input file:
if args.infile != None:
    infile=open(args.infile, "r")
else:
    sys.stderr.write("Please specify inputfile!\n")
    sys.exit(2)

# open output file:
if args.outfile != None:
    outfilename=args.outfile
else:
    sys.stderr.write("Please specify outputfile!\n")
    sys.exit(2)

# open file with MHC pseudo sequences:
if args.mhc != None:
    mhcfilename=args.mhc
else:
    sys.stderr.write("Please specify file with MHC pseudo sequences!\n")
    sys.exit(2)

# open file with BLOSUM matrix:
if args.blosum != None:
    blosumfilename=args.blosum
else:
    sys.stderr.write("Please specify file with BLOSUM matrix!\n")
    sys.exit(2)


##########################################################################
#	READ AND SAVE BLOSUM MATRIX AND MHC PSEUDO SEQUENCE
##########################################################################

blosum = read_blosum(blosumfilename)
mhc,mhc_seq_len = read_MHC_pseudo_seq (mhcfilename)


##########################################################################
#	ENCODE DATA:
##########################################################################

# get dimensions:
enclen=21
n_pep=0
n_pep_aa=0
pep_seqs=[]
mhc_seqs=[]
tmp_targets=[]
for l in infile:
    l=filter(None, l.strip().split())
    # #exclude peptides longer than 20 AA
    # if len(l[0]) <= 20:
    n_pep += 1
    n_pep_aa += len(l[0])
    pep_seqs.append(l[0])
    mhc_seqs.append(mhc[l[2]])
    tmp_targets.append(l[1])
infile.close()

# initialize variables:
peplength=np.zeros((n_pep), dtype=int)
mhclength=np.zeros((n_pep), dtype=int)
peptides=np.zeros((n_pep_aa, enclen), dtype=float)
mhcseqs=np.zeros((n_pep * mhc_seq_len, enclen), dtype=float)
targets=np.zeros((n_pep), dtype=float)

pep_pos=0
mhc_pos=0


# save encoded sequences:
for i in range(0,n_pep):

    #if (str(l[2]) == "DRB1_0101") & (len(l[0]) == 15) & (count<200):
    #print l
    pep_seq=pep_seqs[i]
    mhc_seq=mhc_seqs[i]
    # save peptide and MHC seq length:
    peplength[i] = len(pep_seq)
    mhclength[i] = len(mhc_seq)

    # encode peptide and save:
    for a in pep_seq:
        if a in blosum:
            peptides[pep_pos] = np.array(blosum[a])
            pep_pos +=1
        else:
            sys.stderr.write("Unknown amino acid in peptides, encoding aborted!\n")
            sys.exit(2)

    # ecode MHC pseudo sequence and save:
    for a in mhc_seq:
        if a in blosum:
            mhcseqs[mhc_pos] = np.array(blosum[a])
            mhc_pos += 1
        else:
            sys.stderr.write("Unknown amino acid in MHC, encoding aborted!\n")
            sys.exit(2)

    # save target value (log transformed binding affinity):
    targets[i] = tmp_targets[i]



print("peptides.shape:")
print(peptides.shape)
print("mhcseqs.shape:")
print(mhcseqs.shape)
print("targets.shape:")
print(targets.shape)


# save encoded data as netCDF file:
pep2netcdf(outfilename, peptides, peplength, mhcseqs, mhclength, targets)
