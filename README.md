# PREDICTION OF PROTEIN SUBCELLULAR LOCALIZATION

## Synopsis
Convolutional Bidirectional LSTM with attention mechanism for predicting protein subcellular localization. The model was trained using the  MultiLoc dataset, which counts with 5959 proteins. 

## Author
Jose Juan Almagro Armenteros, DTU Bioinformatics

## Protein data

There are two files in the data folder:
	1) "test.npz" independent set to calculate the final performance of the model
	2) "train.npz" training set.

Each file includes a numpy array with the proteins sequences already encoded in profiles, a numpy array with the masks of each sequence and a numpy vector with the target of each protein.

## Training

The training is performed running the script "train.py". This is a minimal example:

python train.py -i train.npz -t test.npz

The default options are the optimals one, but the training will be really slow on CPU.

To run it on GPU use these flags before the command

THEANO_FLAGS=device=gpu0,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once,warn_float64=warn python train.py -i train.npz -t test.npz

## Contributors
Ole Winther, DTU Compute	
Henrik Nielsen, DTU Bioinformatics	
Søren and Casper Kaae Sønderby, University of Copenhagen
