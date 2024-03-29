# Copyright 2022 Yacouba Kaloga
# SPDX-License-Identifier: Apache-2.0

# Copied and modified from Simple Graph Metric Learning (https://github.com/Yacnnn/SGML/).
# This file is the same as the original one (except some possible changes in the imports paths/names).

import os
import ot
import numpy as np
import scipy.io as sio
import tensorflow as tf 
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tqdm import tqdm 
from .sgcn import Sgcn
from .sgcn_plus import Sgcn_plus
from .gin import Gin
from .graph_attention_network import Graph_attention_network
from .krylov import Krylov
from .process import uniform_transport_matrix
from .process import hammersley_sphere_seq
from .process import dpp_sphere_smpl

class Xw4d(tf.keras.Model):
    def __init__(self, 
                 gcn_type = "krylov-4",
                 store_apxf = False,
                 l2reg = 0,
                 loss_name = "NCA",
                 num_of_layer = 3,
                 hidden_layer_dim = 32,
                 final_layer_dim = 32,
                 nonlinearity = "tanh",
                 gcn_extra_parameters = {},
                 sampling_type = "uniform",
                 num_of_theta_sampled = 1, 
                 dataset = ""
                ):
        super(Xw4d, self).__init__() 
        #Parameters
        self.gcn_type = gcn_type
        self.store_apxf = store_apxf
        self.l2reg = l2reg
        self.loss_name = loss_name
        self.num_of_layer = num_of_layer
        self.hidden_layer_dim = hidden_layer_dim
        self.final_layer_dim = final_layer_dim
        self.nonlinearity = nonlinearity
        self.gcn_extra_parameters = gcn_extra_parameters
        self.sampling_type = sampling_type
        self.num_of_theta_sampled = num_of_theta_sampled
        self.dataset = dataset
        #
        self.transportmatrix = {}

    def build(self,shape):
        self.set_layer_size(shape)
        self.set_gcn(self.nonlinearity)
        self.update_theta()
        
    def set_layer_size(self, shape):
        if self.num_of_layer == 0 :
            self.final_layer_dim = shape[0][1]
        if self.hidden_layer_dim == 0 :
            self.hidden_layer_dim = shape[0][1]
        if self.final_layer_dim == 0 :
            self.final_layer_dim = shape[0][1]
        if self.final_layer_dim == -1 :
            self.final_layer_dim = int(shape[0][1] /2) if shape[0][1] % 2 == 0 else int(np.ceil( shape[0][1] /2 ))
        if self.final_layer_dim == -2 :
            self.final_layer_dim = int(shape[0][1] /4) if shape[0][1] % 4 == 0 else int(np.ceil(  shape[0][1] /4 ))
        if shape[0][1] < 5 :
            self.final_layer_dim = shape[0][1]

    def set_gcn(self, nonlinearity):
        if "sgcn" in self.gcn_type:
            self.gcn = Sgcn( 
                        output_dim=self.final_layer_dim,
                        num_of_layer = self.num_of_layer,
                        nonlinearity = self.nonlinearity,
                        trainable = True,
                        l2reg = self.l2reg,
                        store_apxf=self.store_apxf)
        if "sgcn+" in self.gcn_type:
            self.gcn = Sgcn_plus( 
                        output_dim=self.final_layer_dim,
                        num_of_layer = self.num_of_layer,
                        nonlinearity = self.nonlinearity,
                        trainable = True,
                        l2reg = self.l2reg,
                        store_apxf= self.store_apxf,
                        gcn_extra_parameters = self.gcn_extra_parameters)
        if "gin" in self.gcn_type:
            self.gcn = Gin(self.num_of_layer - 1 , 3, 16, self.final_layer_dim,0 )
        if "gat" in self.gcn_type:
            self.gcn =   Graph_attention_network(
                            nb_of_layer = self.num_of_layer,
                            output_features_size = self.final_layer_dim,
                            learning_rate = 0,
                            dropout_rate = 0,
                            n_attn_heads = 3,
                            l2_reg = self.l2reg)
        if "krylov" in self.gcn_type:
            self.gcn =   Krylov(output_dim = self.final_layer_dim)
            
    def update_theta(self, nb_theta = None):
        if self.sampling_type == "uniform" :
            self.thetalist = tf.linalg.normalize( tf.random.normal([self.final_layer_dim,self.num_of_theta_sampled]) , ord='euclidean', axis=0)[0]
        elif self.sampling_type == "basis" : 
            self.thetalist = tf.eye(self.final_layer_dim) 
        elif self.sampling_type == "ortho" : 
            q = self.num_of_theta_sampled // self.final_layer_dim
            r = self.num_of_theta_sampled - q*self.final_layer_dim
            tampon =  [ tfp.math.gram_schmidt( tf.linalg.normalize( tf.random.normal([self.final_layer_dim, self.final_layer_dim,]) , ord='euclidean', axis=0)[0] ) for k in range(q)]
            if r > 0 :
                tampon.append(tfp.math.gram_schmidt( tf.linalg.normalize( tf.random.normal([self.final_layer_dim, self.final_layer_dim,]) , ord='euclidean', axis=0)[0] )[:,:r])
            self.thetalist = tf.concat(tampon, axis = 1)
        elif self.sampling_type == "dpp" : 
            self.thetalist = dpp_sphere_smpl( n = self.num_of_theta_sampled, p = self.final_layer_dim)
        elif self.sampling_type == "hamm" :
            self.thetalist = hammersley_sphere_seq( n = self.num_of_theta_sampled, p = self.final_layer_dim)

    @tf.custom_gradient
    def grad_reverse(self,x):
        y = tf.identity(x)
        def custom_grad(dy):
            return -dy
        return y, custom_grad

    def loss(self,distances_sq, labels = []):
        if self.loss_name == "regular" :
            return self.grad_reverse( tf.reduce_sum(distances_sq) )
        elif self.loss_name == "NCA" :
            return self.grad_reverse(self.NCA(distances_sq , labels))
        elif self.loss_name == "NCCML" :
            return self.grad_reverse(self.NCCML(distances_sq , labels))
        elif "LMNN" in self.loss_name :
            return self.grad_reverse(self.LMNN(distances_sq , labels , k = int(self.loss_name.split("-")[-1]) ))
        return None

    def NCA(self,distances_sq, labels):
        n = tf.shape(distances_sq)[0]
        cross_labels = tf.cast( labels - tf.transpose(labels) == 0, tf.float32)
        expmd =  tf.math.exp( - distances_sq)* tf.cast((tf.ones([n,n]) - tf.eye(n)),tf.float32)
        s = tf.reduce_sum( expmd , axis = 1)[:,np.newaxis]
        p = expmd  / s
        p = tf.reduce_sum( p*cross_labels , axis = 1)
        return tf.reduce_sum(p)

    def NCCML(self,distances_sq, labels):
        n = tf.shape(distances_sq)[0]
        cross_labels = tf.cast( labels - tf.transpose(labels) == 0, tf.float32)
        labels_weight =  tf.cast( tf.ones((n,1)), tf.float32) / tf.reduce_sum(cross_labels, axis = 1)[:,np.newaxis]
        expmd = tf.math.exp( - (1.0/2) * distances_sq @ (cross_labels  * labels_weight) ) 
        s = expmd  @ ( labels_weight)
        p = tf.linalg.diag_part(expmd)[:,np.newaxis] / s
        return tf.reduce_sum(tf.math.log(p))

    def LMNN(self, distances_sq, labels, k , mu = 1.0 / 2):
        n = tf.shape(distances_sq)[0]
        cross_labels = tf.cast( labels - tf.transpose(labels) == 0, tf.float32) * tf.cast((tf.ones([n,n]) - tf.eye(n)),tf.float32)
        argsort_indices = tf.argsort( distances_sq, axis=-1, direction='ASCENDING')
        pull_mask = tf.gather(cross_labels , argsort_indices, batch_dims=-1)
        sort_distances_sq = tf.gather(distances_sq, argsort_indices, batch_dims=-1)
        pull_distances = sort_distances_sq[:,:k+1]*pull_mask[:,:k+1]
        pull = tf.reduce_sum(pull_distances)
        push = 0
        cross_labelsv2 = tf.cast( labels - tf.transpose(labels) == 0, tf.float32)
        for i in range(n):
            push += tf.reduce_sum(   tf.nn.relu(   (pull_distances[i:i+1,:] -  distances_sq[:,i:i+1]  + 1)*pull_mask[i,:k+1]*(1 - cross_labelsv2[:,i:i+1])    )     )
        return  (1-mu)*pull + mu*push

    def build_transport_matrix(self, feat):
        file_name = self.dataset + "_transport_matrix.mat"
        if os.path.isfile('precomp/'+file_name):
            self.transportmatrix = sio.loadmat('precomp/'+file_name)
        else :
            for i in tqdm(range(len(feat)), unit = 'trprtmtrx'):
                for j in range(len(feat)):
                    if i >= j :
                        key = str(feat[i].shape[0]) + "-" + str(feat[j].shape[0])
                        keyrev = str(feat[j].shape[0]) + "-" + str(feat[i].shape[0])
                        if key not in self.transportmatrix.keys(): 
                            self.transportmatrix[key] = uniform_transport_matrix(feat[i].shape[0],feat[j].shape[0])
                            self.transportmatrix[keyrev] = np.transpose(self.transportmatrix[key])
            sio.savemat('precomp/'+file_name, mdict=self.transportmatrix)
            self.themax = max([len(f) for f in feat])
                        
    def build_tm(self, feat, themax):
        tm = []
        for i in tqdm(range(len(feat)), unit = 'trprtmtrx', disable=True):
            for j in range(len(feat)):
                # if i == j :
                #    tm.append(  tf.zeros((themax,themax)) )
                if j >= i :
                    key = str(feat[i].shape[0]) + "-" + str(feat[j].shape[0])
                    a = tf.concat( [self.transportmatrix[key], tf.zeros([themax-feat[i].shape[0],feat[j].shape[0]])] , axis = 0) 
                    b = tf.concat( [a, tf.zeros([themax,themax-feat[j].shape[0]])] , axis = 1) 
                    tm.append(  b )
        return tf.convert_to_tensor(tm)

    def build_tmv2(self, feat, feat2, themax):
            tm = []
            for i in tqdm(range(len(feat)), unit = 'trprtmtrx', disable=True):
                for j in range(len(feat2)):
                    # if i == j :
                    #    tm.append(  tf.zeros((themax,themax)) )
                    # if j >= i :
                    key = str(feat[i].shape[0]) + "-" + str(feat2[j].shape[0])
                    a = tf.concat( [self.transportmatrix[key], tf.zeros([themax-feat[i].shape[0],feat2[j].shape[0]])] , axis = 0) 
                    b = tf.concat( [a, tf.zeros([themax,themax-feat2[j].shape[0]])] , axis = 1) 
                    tm.append(  b )
            return tf.convert_to_tensor(tm)

    def build_on_tm(self, feat, themax):
        tm = []
        for i in tqdm(range(len(feat)), unit = 'trprtmtrx', disable=True):
            for j in range(len(feat)):
                # if i == j :
                #    tm.append(  tf.zeros((themax,themax)) )
                if j >= i :
                    transportmatrix = uniform_transport_matrix(feat[i].shape[0],feat[j].shape[0])
                    key = str(feat[i].shape[0]) + "-" + str(feat[j].shape[0])
                    a = tf.concat( [transportmatrix, tf.zeros([themax-feat[i].shape[0],feat[j].shape[0]])] , axis = 0) 
                    b = tf.concat( [a, tf.zeros([themax,themax-feat[j].shape[0]])] , axis = 1) 
                    tm.append(  b )
        return tf.convert_to_tensor(tm)

    def build_on_tmv2(self, feat, feat2, themax):
            tm = []
            for i in tqdm(range(len(feat)), unit = 'trprtmtrx', disable=True):
                for j in range(len(feat2)):
                    # if i == j :
                    #    tm.append(  tf.zeros((themax,themax)) )
                    # if j >= i :
                    transportmatrix = uniform_transport_matrix(feat[i].shape[0],feat2[j].shape[0])
                    key = str(feat[i].shape[0]) + "-" + str(feat2[j].shape[0])
                    a = tf.concat( [transportmatrix, tf.zeros([themax-feat[i].shape[0],feat2[j].shape[0]])] , axis = 0) 
                    b = tf.concat( [a, tf.zeros([themax,themax-feat2[j].shape[0]])] , axis = 1) 
                    tm.append(  b )
            return tf.convert_to_tensor(tm)