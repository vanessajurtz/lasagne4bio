# Authors:
#	Vanessa Isabell Jurtz, DTU Bioinformatics
#	Morten Nielsen, DTU Bioinformatics

################################################################################
#	PEPTIDE BINDING TO MHCII
################################################################################

1) encode data

2) train networks

3) make ensemble predictions

################################################################################
#	ENCODE DATA
################################################################################

Amino acid sequences are encoded using the BLOSUM 62 matrix. The dataset is
partitioned into 5 subsets for cross-validation. Note that the subsets need to
combined accordingly to perform nested or non-nested cross-validation.

python scripts/seq2bl.py \
    -i data/c000 \
    -o data/c000.bl.nc \
    -m data/pseudosequences.all.X.dat \
    -b data/BLOSUM62.txt

python scripts/seq2bl.py \
    -i data/c001 \
    -o data/c001.bl.nc \
    -m data/pseudosequences.all.X.dat \
    -b data/BLOSUM62.txt

python scripts/seq2bl.py \
    -i data/c002 \
    -o data/c002.bl.nc \
    -m data/pseudosequences.all.X.dat \
    -b data/BLOSUM62.txt

python scripts/seq2bl.py \
    -i data/c003 \
    -o data/c003.bl.nc \
    -m data/pseudosequences.all.X.dat \
    -b data/BLOSUM62.txt

python scripts/seq2bl.py \
    -i data/c004 \
    -o data/c004.bl.nc \
    -m data/pseudosequences.all.X.dat \
    -b data/BLOSUM62.txt

################################################################################
#	TRAIN NETWORKS
################################################################################

The scripts to train neural networks have many command line options to set
various parameters. Please have a look at the scripts to see them. Below you can
find a minmal example to test if the code is running, it's parameter settings do
not give optimal performance.

python scripts/cnn_train.py \
    -training_data data/c000.bl.nc \
    -validation_data data/c001.bl.nc \
    -model_out cnn_params_1.npz

python scripts/cnn_train.py \
    -training_data data/c002.bl.nc \
    -validation_data data/c003.bl.nc \
    -model_out cnn_params_2.npz

python scripts/lstm_train.py \
    -training_data data/c000.bl.nc \
    -validation_data data/c001.bl.nc \
    -model_out lstm_params_1.npz

python scripts/lstm_train.py \
    -training_data data/c002.bl.nc \
    -validation_data data/c003.bl.nc \
    -model_out lstm_params_2.npz



################################################################################
#	PREDICT NETWORKS
################################################################################

To predict networks you need to make a list of all networks in the ensemble you
want to predcit:

echo "cnn_params_1.npz" > ensemble_list_cnn.txt
echo "cnn_params_2.npz" >> ensemble_list_cnn.txt

echo "lstm_params_1.npz" > ensemble_list_lstm.txt
echo "lstm_params_2.npz" >> ensemble_list_lstm.txt

Then predict the networks:

python scripts/cnn_ensemble.py \
    -data data/c004.bl.nc \
    -data_aa data/c004 \
    -ensemblelist ensemble_list_cnn.txt \
    -out cnn_pred.txt

python scripts/lstm_ensemble.py \
    -data data/c004.bl.nc \
    -data_aa data/c004 \
    -ensemblelist ensemble_list_lstm.txt \
    -out lstm_pred.txt

Combine ensembles with different network architecture:

First make a list of the pred files and how they should be weighted:
echo -e "cnn_pred.txt\t1" > ensemble_list.txt
echo -e "lstm_pred.txt\t1" >> ensemble_list.txt

run combine script:
python scripts/combine_predictions.py \
    -infile ensemble_list.txt \
    -out all_pred.txt
