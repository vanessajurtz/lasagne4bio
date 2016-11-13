import sys
import numpy as np
import importlib
import lasagne as nn
import theano
from theano import tensor as T
import os
import glob

import data
import utils

if not (2 <= len(sys.argv) <= 3):
    sys.exit("Usage: python predict.py <metadata_path> [subset=test]")

sym_y = T.imatrix('target_output')
sym_x = T.tensor3()

metadata_path_all = glob.glob(sys.argv[1] + "*")

print "shape of metadata_path_all"
print(len(metadata_path_all))

if len(sys.argv) >= 3:
    subset = sys.argv[2]
    assert subset in ['train', 'valid', 'test', 'train_valid']
else:
    subset = 'test'

if subset == "test":
    X, mask, _, num_seq = data.get_test()
elif subset == "train":
    sys.exit("train not implemented")
elif subset == "train_valid":
    sys.exit("train_valid not implemented")
else:
    sys.exit("valid not implemented")


for metadata_path in metadata_path_all:

    print "Loading metadata file %s" % metadata_path

    metadata = np.load(metadata_path)

    config_name = metadata['config_name']

    config = importlib.import_module("configurations.%s" % config_name)

    print "Using configurations: '%s'" % config_name

    print "Build model"

    l_in, l_out = config.build_model()

    print "Build eval function"

    inference = nn.layers.get_output(
        l_out, sym_x, deterministic=True)

    print "Load parameters"

    nn.layers.set_all_param_values(l_out, metadata['param_values'])

    print "Compile functions"

    predict = theano.function([sym_x], inference)

    print "Predict"

    predictions = []
    batch_size = config.batch_size
    num_batches = np.size(X,axis=0) // batch_size

    for i in range(num_batches):
        idx = range(i*batch_size, (i+1)*batch_size)
        x_batch = X[idx]
        mask_batch = mask[idx]
        p = predict(x_batch)
        predictions.append(p)
        
    predictions = np.concatenate(predictions, axis = 0)
    predictions_path = os.path.join("predictions", os.path.basename(metadata_path).replace("dump_", "predictions_").replace(".pkl", ".npy"))
    
    print "Storing predictions in %s" % predictions_path
    np.save(predictions_path, predictions)
