import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
import string
import sys
from datetime import datetime, timedelta
import importlib
import time
import cPickle as pickle
print ("loading data ...")
#import data
print ("loading data completed ...")
import utils

np.random.seed(1)

if len(sys.argv) != 2:
    sys.exit("Usage: python train.py <config_name>")

config_name = sys.argv[1]

#config_name = "lstm_uni_20"
config = importlib.import_module("configurations.%s" % config_name)
optimizer = config.optimizer
print "Using configurations: '%s'" % config_name

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s" % (config_name, timestamp)
metadata_path = "metadata/dump_%s" % experiment_id

print "Experiment id: %s" % experiment_id

#print "Build model"


# Min/max sequence length
# Number of units in the hidden (recurrent) layer
#N_HIDDEN = 10
# Number of training sequences in each batch
# Optimization learning rate
#LEARNING_RATE = .001
# All gradients above this will be clipped
#GRAD_CLIP = 100

num_classes = 8


def main():
    sym_y = T.imatrix('target_output')
    sym_mask = T.matrix('mask')
    sym_x = T.tensor3()

    TOL = 1e-5
    num_epochs = config.epochs
    batch_size = config.batch_size

#### DATA ####
#    print "@@@@TESTING@@@@"
#    l_in = nn.layers.InputLayer(shape=(None, 700, 42))
#    l_dim_a = nn.layers.DimshuffleLayer(
#        l_in, (0,2,1))
#    l_conv_a = nn.layers.Conv1DLayer(
#        incoming=l_dim_a, num_filters=42, border_mode='same',
#        filter_size=3, stride=1, nonlinearity=nn.nonlinearities.rectify)
#    l_dim_b = nn.layers.DimshuffleLayer(
#        l_conv_a, (0,2,1))
#    out = nn.layers.get_output(l_dim_b, sym_x)
#    testvar = np.ones((128, 700, 42)).astype('float32')
#    print "@@@@EVAL@@@@"
#    john = out.eval({sym_x: testvar})
#    print("Johns shape")
#    print(john.shape)


    print("Building network ...")
    ##########################DEBUG##########################
    l_in, l_out = config.build_model()
    
    ##########################DEBUG##########################
    all_layers = nn.layers.get_all_layers(l_out)
    num_params = nn.layers.count_params(l_out)
    print("  number of parameters: %d" % num_params)
    print("  layer output shapes:")
    for layer in all_layers:
        name = string.ljust(layer.__class__.__name__, 32)
        print("    %s %s" % (name, nn.layers.get_output_shape(layer)))
    print("Creating cost function")
    # lasagne.layers.get_output produces a variable for the output of the net
    out_train = nn.layers.get_output(
        l_out, sym_x, deterministic=False)

#    testvar = np.ones((128, 700, 42)).astype('float32')
#    john = out_train.eval({sym_x: testvar})
#    print("@@@@@JOHN@@@@@")
#    print(john.shape)
#    print(john.reshape((-1, num_classes)).shape)

    print("Creating eval function")
    out_eval = nn.layers.get_output(
        l_out, sym_x, deterministic=True)

    probs_flat = out_train.reshape((-1, num_classes))

    lambda_reg = config.lambda_reg
    params = nn.layers.get_all_params(l_out, regularizable=True)
    reg_term = sum(T.sum(p**2) for p in params)
    cost = T.nnet.categorical_crossentropy(T.clip(probs_flat, TOL, 1-TOL), sym_y.flatten())
    cost = T.sum(cost*sym_mask.flatten()) / T.sum(sym_mask) + lambda_reg * reg_term

    # Retrieve all parameters from the network
    all_params = nn.layers.get_all_params(l_out, trainable=True)
    # Setting the weights
    if hasattr(config, 'set_weights'):
        nn.layers.set_all_param_values(l_out, config.set_weights())
    # Compute SGD updates for training
    print("Computing updates ...")
    if hasattr(config, 'learning_rate_schedule'):
        learning_rate_schedule = config.learning_rate_schedule              # Import learning rate schedule
    else:
        learning_rate_schedule = { 0: config.learning_rate }
    learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

    all_grads = T.grad(cost, all_params)

    cut_norm = config.cut_grad
    updates, norm_calc = nn.updates.total_norm_constraint(all_grads, max_norm=cut_norm, return_norm=True)

    if optimizer == "rmsprop":
        updates = nn.updates.rmsprop(updates, all_params, learning_rate)
    elif optimizer == "adadelta":
        updates = nn.updates.adadelta(updates, all_params, learning_rate)
    elif optimizer == "adagrad":
        updates = nn.updates.adagrad(updates, all_params, learning_rate)
    elif optimizer == "nag":
        momentum_schedule = config.momentum_schedule
        momentum = theano.shared(np.float32(momentum_schedule[0]))
        updates = nn.updates.nesterov_momentum(updates, all_params, learning_rate, momentum)
    else:
        sys.exit("please choose either <rmsprop/adagrad/adadelta/nag> in configfile")
            
    # Theano functions for training and computing cost
    print "config.batch_size %d" %batch_size
    print "data.num_classes %d" %num_classes
    if hasattr(config, 'build_model'):
        print("has build model")
    print("Compiling train ...")
    # Use this for training (see deterministic = False above)
    train = theano.function(
        [sym_x, sym_y, sym_mask], [cost, out_train, norm_calc], updates=updates)

    print("Compiling eval ...")
    # use this for eval (deterministic = True + no updates)
    eval = theano.function([sym_x, sym_y, sym_mask], [cost, out_eval])

    # Start timers
    start_time = time.time()
    prev_time = start_time

    all_losses_train = []
    all_accuracy_train = []
    all_losses_eval_train = []
    all_losses_eval_valid = []
    all_losses_eval_test = []
    all_accuracy_eval_train = []
    all_accuracy_eval_valid = []
    all_accuracy_eval_test = []
    all_mean_norm = []


    import data
    X_train, X_valid, y_train, y_valid, mask_train, mask_valid, num_seq_train \
			= data.get_train()
    print("y shape")
    print(y_valid.shape)
    print("X shape")
    print(X_valid.shape)
    # Start training

    for epoch in range(num_epochs):

        if (epoch % 10) == 0:
            print "Epoch %d of %d" % (epoch + 1, num_epochs)

        if epoch in learning_rate_schedule:
            lr = np.float32(learning_rate_schedule[epoch])
            print "  setting learning rate to %.7f" % lr
            learning_rate.set_value(lr)
        if optimizer == "nag":
            if epoch in momentum_schedule:
                mu = np.float32(momentum_schedule[epoch])
                print "  setting learning rate to %.7f" % mu
                momentum.set_value(mu)
#        print "Shuffling data"
        seq_names = np.arange(0,num_seq_train)
        np.random.shuffle(seq_names)     
        X_train = X_train[seq_names]
        y_train = y_train[seq_names]
        mask_train = mask_train[seq_names]

        num_batches = num_seq_train // batch_size
        losses = []
        preds = []
        norms = []
        for i in range(num_batches):
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_train[idx]
            y_batch = y_train[idx]
            mask_batch = mask_train[idx]
            loss, out, batch_norm = train(x_batch, y_batch, mask_batch)
#            print(batch_norm)
            norms.append(batch_norm)
            preds.append(out)
            losses.append(loss)

#            if ((i+1) % config.write_every_batch == 0) | (i == 0):
#                if i == 0:
#                    start_place = 0
#                else:
#                    start_place = i-config.write_every_batch
#                print "Batch %d of %d" % (i + 1, num_batches)
#                print "  curbatch training loss: %.5f" % np.mean(losses[start_place:(i+1)])
#                print "  curbatch training acc: %.5f" % np.mean(accuracy[start_place:(i+1)])
        predictions = np.concatenate(preds, axis = 0)
        loss_train = np.mean(losses)
        all_losses_train.append(loss_train)

        acc_train = utils.proteins_acc(predictions, y_train[0:num_batches*batch_size], mask_train[0:num_batches*batch_size])
        all_accuracy_train.append(acc_train)

        mean_norm = np.mean(norms)
        all_mean_norm.append(mean_norm)

        if 1==1:
            print "  average training loss: %.5f" % loss_train
            print "  average training accuracy: %.5f" % acc_train
            print "  average norm: %.5f" % mean_norm

            sets = [#('train', X_train, y_train, mask_train, all_losses_eval_train, all_accuracy_eval_train),
                    ('valid', X_valid, y_valid, mask_valid, all_losses_eval_valid, all_accuracy_eval_valid)]
            for subset, X, y, mask, all_losses, all_accuracy in sets:
                print "  validating: %s loss" % subset
                preds = []
                num_batches = np.size(X,axis=0) // config.batch_size
                for i in range(num_batches): ## +1 to get the "rest"
#                    print(i)
                    idx = range(i*batch_size, (i+1)*batch_size)
                    x_batch = X[idx]
                    y_batch = y[idx]
                    mask_batch = mask[idx]
                    loss, out = eval(x_batch, y_batch, mask_batch)
                    preds.append(out)
#                    acc = utils.proteins_acc(out, y_batch, mask_batch)
                    losses.append(loss)
#                    accuracy.append(acc)
                predictions = np.concatenate(preds, axis = 0)
#                print "  pred"
#                print(predictions.shape)
#                print(predictions.dtype)
                loss_eval = np.mean(losses)
                all_losses.append(loss_eval)

#                acc_eval = np.mean(accuracy)
                acc_eval = utils.proteins_acc(predictions, y, mask)
                all_accuracy.append(acc_eval)

                print "  average evaluation loss (%s): %.5f" % (subset, loss_eval)
                print "  average evaluation accuracy (%s): %.5f" % (subset, acc_eval)

        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * num_epochs
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
        print

        if (epoch >= config.start_saving_at) and ((epoch % config.save_every) == 0):
            print "  saving parameters and metadata"
            with open((metadata_path + "-%d" % (epoch) + ".pkl"), 'w') as f:
                pickle.dump({
                        'config_name': config_name,
                        'param_values': nn.layers.get_all_param_values(l_out),
                        'losses_train': all_losses_train,
                        'accuracy_train': all_accuracy_train,
                        'losses_eval_train': all_losses_eval_train,
                        'losses_eval_valid': all_losses_eval_valid,
			'losses_eval_test': all_losses_eval_test,
                        'accuracy_eval_valid': all_accuracy_eval_valid,
                        'accuracy_eval_train': all_accuracy_eval_train,
			'accuracy_eval_test': all_accuracy_eval_test,
                        'mean_norm' : all_mean_norm,
                        'time_since_start': time_since_start,
                        'i': i,
                    }, f, pickle.HIGHEST_PROTOCOL)

            print "  stored in %s" % metadata_path

    print

if __name__ == '__main__':
    main()
