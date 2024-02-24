# Copyright 2022 Yacouba Kaloga
# SPDX-License-Identifier: Apache-2.0

# Copied and modified from Simple Graph Metric Learning (https://github.com/Yacnnn/SGML/).
# This file gathers functions needed to compute distance matrices using SGML code.
# We reused the code from their repository to compute the distance matrices using SGML but added our functions to split 
# and load datasets (for a fair comparison).

from sklearn.model_selection import KFold, StratifiedKFold
import itertools
import numpy as np
import random
import scipy.io as sio
import ast
import os
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from sklearn.model_selection import ParameterGrid
from .process_data import load_dataset
from .pw4d import Pw4d
import time

# This function is a copy from graph_gp/splitters.py to use the exact same splits we testing the sgml kernel.
def split_train_test_indices_stratified_fold(n_splits: int, random_state: int, i: int, indices: np.ndarray, y: np.ndarray=None):
    """
    This function perform a stratified K-fold of the indices according to y using n_splits and returns the i-th split. 
    Uses a specific random state.
    Taking i in range 0, ..., n_splits-1 will give all the succesive train-test folds.
    If a stratified K-fold is impossible or y is not specified, uses a K-fold instead.
    
    Args:
        n_splits: The number of splits of the K-fold.
        random_state: The random state that determines the n_split folds.
        i: The number of the fold to return. i should be less than n_splits
        indices: The list of indices to split.
        y: The output scalars used to startify the splits.
    Returns:
        indices_train: The train indices of the i-th fold.
        indices_test: The test (or valid) indices of the i-th fold.
    """
    if y is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    try:
        indices_train, indices_test = next(itertools.islice(splitter.split(indices,y), i, None))
    except ValueError:
        print(f"Caution, n_splits={n_splits} cannot be greater than the number of members in each class. Using a KFold instead.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        indices_train, indices_test = next(itertools.islice(splitter.split(indices,indices), i, None))
    return indices_train, indices_test


MAX_RESTART = 1
def create_batch(features, structures, labels, batch_size = 32, shuffle = True, precomputed_batch = False):
    """ Creates a list of batch of data. """
    X = np.copy(features)
    W = np.copy(structures)
    labels = np.copy(labels)
    n = X.shape[0]
    if shuffle:
        s = np.arange(n) 
        np.random.shuffle(s)
        X = features[s]  
        W = W[s]
        labels = labels[s]
    q = n//batch_size
    block_end = q*batch_size   
    batch_limit = [ [k*batch_size,(k+1)*batch_size]  for k in range(q)] + [[q*batch_size,n]]
    if batch_limit[-1][1]  - batch_limit[-1][0]  == 1 :
        batch_limit[-2][1] =  batch_limit[-1][1]
        batch_limit = batch_limit[:-1]
    batch_features = [ X[ind[0]:ind[1]] for ind in batch_limit]
    batch_structures = [ W[ind[0]:ind[1]] for ind in batch_limit]
    batch_labels = [ labels[ind[0]:ind[1]] for ind in batch_limit]
    batch_index = [  s[ind[0]:ind[1]]  for ind in batch_limit]#.astype(np.float32)
    if batch_features[-1].shape[0] == 0 :
        return batch_features[:-1], batch_structures[:-1], batch_labels[:-1], batch_index[:-1]
    return batch_features, batch_structures, batch_labels, batch_index

def compute_all_dataset_distances_sgml(dataset_name, nums_of_layers, feature, config_params, seed=0, verbose=0):
    device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'
    print("Device:", device)
    # feature in ["attributes", "fuse"]
    
    root_datasets = config_params["datasets"]["root_TUDatasets"]
    start_time_dataset = time.time()
    data = load_dataset(dataset_name, feature=feature, h=0, root_datasets = root_datasets)
    time_dataset = time.time()-start_time_dataset
    y = data["labels"]
    
    parameters = {}
    parameters["num_of_layer"] = nums_of_layers
    parameters["num_of_iter"] = [10]
    parameters["batch_size"] = [8]
    parameters["learning_rate"] = [0.999e-2]
    if dataset_name == "ENZYMES":
        parameters["learning_rate"] = [0.999e-3]
        parameters["num_of_iter"] = [20] 
    if dataset_name == "PROTEINS":
        parameters["learning_rate"] = [0.999e-4]
        parameters["num_of_iter"] = [20] 
    if dataset_name == "COX2" or dataset_name == "BZR":
        parameters["batch_size"] = [64]
        
    total_time_embeddings, total_time_distances = 0., 0.
    all_D_matrices = []
    
    list_of_parameters = list(ParameterGrid(parameters))
    indices_train, indices_test = split_train_test_indices_stratified_fold(config_params["datasets"]["classification_NSPLITS"], config_params["datasets"]["classification_stratified_split_seed"], seed, np.arange(len(y)), y=y)
    for parameters_ in list_of_parameters:
        with tf.device(device):
            distances, times_embeddings, times_distances = distance_matrices_sgml(data, indices_train, indices_test, parameters_, seed=seed, verbose=verbose)
        total_time_embeddings += times_embeddings
        total_time_distances += times_distances
        all_D_matrices.append(distances)
    return all_D_matrices, time_dataset, total_time_embeddings, total_time_distances

def distance_matrices_sgml(data, indices_train, indices_test, parameters, seed=0, verbose=0):
    """ Train the model given multiviews data and specified parameters and return it. """    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = str(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)

    features_train = data["features"][indices_train]
    structures_train = data["structures"][indices_train]
    y_train = data["labels"][indices_train]
    
    # Training parameters
    learning_rate = parameters["learning_rate"]
    num_of_iter = parameters["num_of_iter"]
    batch_size = parameters["batch_size"]
    num_of_layer = parameters["num_of_layer"]
    decay_learning_rate = False
    
    Xw4d = Pw4d
    model_xw4d = Xw4d( 
                        gcn_type = "sgcn",
                        l2reg = 0,
                        loss_name = "NCCML",
                        num_of_layer = num_of_layer,
                        hidden_layer_dim = 0,
                        final_layer_dim = 5,
                        nonlinearity = "relu",
                        store_apxf = False,
                        gcn_extra_parameters = {},
                        sampling_type = "uniform", 
                        num_of_theta_sampled = 50,
                    )

    start_time_embeddings = time.time()
    for e in range(num_of_iter):
        batch_features, batch_structures, batch_y, batch_index = create_batch(features_train, structures_train, y_train, batch_size = batch_size, shuffle = True)
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        acc_loss_d = 0
        for feat, struct, lab, ind in zip( batch_features, batch_structures, batch_y, batch_index):
            with tf.GradientTape() as tape:
                loss_d, loss_s =  model_xw4d(list(feat),list(struct),list(lab),list(ind))
                # print(loss_d)
            gradients = tape.gradient(loss_d, model_xw4d.trainable_variables)
            gradient_variables = zip( gradients, model_xw4d.trainable_variables)
            optimizer.apply_gradients(gradient_variables)
            acc_loss_d += loss_d/len(batch_index)
        if decay_learning_rate :
            optimizer.learning_rate = learning_rate * np.math.pow(1.1, - 50.*(e / num_of_iter))
        if verbose>1:
            print(f"Iter {e}")
            print("epochs : " + str(e) + "/" + str(num_of_iter))
            print("avg_loss_d : ", str(acc_loss_d) )
            
    time_embeddings = time.time() - start_time_embeddings
    
    start_time_distances = time.time()
    D = model_xw4d.distance_quad_np(list(data["features"]),list(data["structures"]), list(np.arange(data["features"].shape[0])), display=False)
    time_distances = time.time() - start_time_distances
     
    #save_distance(distance = D, parameters = parameters, title_extension= "_iter"+str(e + 1))
    return [D], time_embeddings, time_distances