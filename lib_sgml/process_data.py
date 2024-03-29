# Copyright 2022 Yacouba Kaloga
# SPDX-License-Identifier: Apache-2.0

# Copied and modified from Simple Graph Metric Learning (https://github.com/Yacnnn/SGML/).
# This file is the same as the original one (except some possible changes in the imports paths/names).

import os
import sys
import copy
import re
import numpy as np
import tensorflow as tf
import igraph as ig
import scipy.io as sio 
import matplotlib.pyplot as plt
import networkx as nx

from pathlib import Path
from tqdm import tqdm
from typing import List
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.base import TransformerMixin
from collections import defaultdict

strpath = lambda x: str(Path(x).resolve())
ROOTDIR = ""
def available_tasks():
    """ Return list of available tasks. """
    return ["sw4d","pw4d","wwl","swwl","fgw"]

def available_datasets():
    """ Return list of available datasets. """
    return ["Alkane","ENZYMES","NCI109","NCI1","PTC_MR","PTC_FR","MUTAG","PROTEINS_full","PROTEINS","BZR","COX2","Cuneiform","IMDB-MULTI", "IMDB-BINARY","SYNTHETIC"]

def available_features():
    """ Return list of available datasets. """
    return ["node_labels","attributes","degree","fuse"]

def available_features_per_datasets(dataset):
    """ Return list of available feature for a given dataset. """
    if dataset in ["NCI109","NCI1","PTC_MR","PTC_FR","MUTAG"]:
        return ["node_labels", "degree"]
    if dataset in ["ENZYMES","PROTEINS_full","PROTEINS","BZR","COX2","Cuneiform","SYNTHETIC"]:
        return ["attributes", "node_labels", "fuse", "degree"]
    if dataset in ["Alkane"]:
        return ["attributes", "degree"]
    if dataset in ["IMDB-MULTI", "IMDB-BINARY"]:
        return ["degree"]
       
 
def load_scalars(dataset, root_datasets=""):
    graph_labels = []
    nb_graph = 0
    root = os.path.join(root_datasets, dataset+'/raw/')
    
    with open(os.path.join(root,dataset+'_graph_labels.txt'), 'r') as f:
        lines = f.readlines()
        graph_labels = [ int(x.strip()) for x in lines]
     
    tampon = {}
    new_graph_labels = []
    k = -1
    for l in graph_labels:
        if l not in tampon:
            k = k + 1
            tampon[l] = k
        new_graph_labels.append(tampon[l])
    return np.array(new_graph_labels).reshape((-1,1))
    
def load_dataset(dataset, feature = "attributes", h = 0, root_datasets=""):
    """ Dataset : - Alkane
                    https://brunl01.users.greyc.fr/CHEMISTRY/
                  - ENZYMES,DD,MUTAG,PROYEINS_full,PTC_MR,NCI109,NCI1,"BZR","COX2","Cuneiform"
                    https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

        Feature :   - node_labels [node labels for dortmund datasets]
                    - attributes [attributes for dortmund datasets, 
                                                 node positions for alkane]
                    - graph_fuse [concatenation of node_labels and features for dortmund datasets]
                    - degree [one hot encoding of degree]

        if h > 0:  - If the features selectionned is continuous, GIN wth h layer is applied and returned as node features
                    - If the features selectionned is discrete, WL wth h layer is applied and returned as node features
                    (see WWL papers)
    """
    if feature not in available_features_per_datasets(dataset):
        print("No "+feature+" in "+dataset)
        sys.exit()
    if dataset in ["NCI109","NCI1","PTC_MR","PTC_FR","MUTAG","DD","ENZYMES","PROTEINS_full","PROTEINS","BZR","COX2","Cuneiform","IMDB-MULTI", "IMDB-BINARY","SYNTHETIC"]:
        data = load_drtmnd_dataset(dataset, use_attributes_if_exist = True, root_datasets=root_datasets) 
        feature_key = "node_labels" if feature == "node_labels" else "graph_features" if feature == "attributes" else "graph_fuse" if feature == "fuse" else "graph_degree"
    elif dataset in ["Alkane"]:
        data = load_alkane()
        feature_key =  "node_positions" if feature == "attributes" else "graph_degree"
    datam = {}
    datam["features"] = data[feature_key]
    datam["structures"] = np.array([d+0.0 for d in data["adjency_matrix"]], dtype = object) 
    datam["labels"] = data["graph_labels"]
    if h > 0 :
        if feature == "features" or feature == 'graph_fuse':
            datam["features"] = create_labels_seq_cont(datam["features"], list(datam["structures"]), h)
        if feature == "node_labels" or feature == "degree":
            datam["features"] = create_labels_seq_cont(datam["features"], list(datam["structures"]), h)
    return datam

def get_labels(dataset_name):
    """ Return the label of specified dataset (When possible). """
    if dataset_name in ["NCI109","NCI1","PTC_MR","MUTAG","IMDB-BINARY","IMDB-MULTI"]:
        data = load_drtmnd_dataset(dataset_name, use_attributes_if_exist = True) 
    elif dataset_name in ["PROTEINS_full","PROTEINS","ENZYMES","BZR","COX2","Cuneiform","SYNTHETIC"]:
        data = load_drtmnd_dataset(dataset_name, use_attributes_if_exist = True) 
    if dataset_name in ["alkane"]:
        data = load_alkane()
    return data['graph_labels']

def normalize_features(X):
    """ X - MEAN(X) / STD(X) """
    mean = np.mean(X, axis=0) 
    std = np.std(X, axis = 0) 
    std[ np.where(std == 0) ] = 1
    X =  (X - mean)/std
    return X

def create_save_rootfolder(task, NOW):
    """ Create root folder results/task for save informations about MVGCCA training. """
    if not os.path.isdir(strpath(ROOTDIR+'results')):
        os.system('mkdir '+strpath(ROOTDIR+'results'))
    if not os.path.isdir(strpath(ROOTDIR+'results/'+task)):
        os.system('mkdir '+strpath(ROOTDIR+'results/'+task))
    if not os.path.isdir(strpath(ROOTDIR+'results/'+task+'/'+NOW)):
        os.system('mkdir '+strpath(ROOTDIR+'results/'+task+'/'+NOW)) 
   
def update_path(parameters, task, NOW, parameters_id, run_id):
    """ Change the path where we save informations about current run. """
    task_ = task#parameters["dataet"]#+"/"+tassk 
    any_write = parameters["write_weights"] or parameters["write_loss"] or parameters["write_latent_space"] or parameters["evaluation"]
    if any_write:
        create_save_rootfolder(task_, NOW)
        os.makedirs(strpath(ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)), exist_ok=True)
        os.makedirs(strpath(ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)), exist_ok=True)
        # if  not os.path.isdir(strpath(ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id))):
        #     os.system('mkdir '+strpath(ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)))
        # if  not os.path.isdir(strpath(ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id))):
        #     os.system('mkdir '+strpath(ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)))
        sio.savemat(ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)+'/parameters_dict.mat',parameters)
    parameters["parameters_main_path"] = ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)
    parameters["weights_path"] = ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/weights/' if parameters["write_weights"] else ''
    parameters["write_loss_path"] = ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/logs/' if parameters["write_loss"] else ''
    parameters["latent_space_path"] = ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/embeddings/' if parameters["write_latent_space"] else ''
    parameters["evaluation_path"] = ROOTDIR+'results/'+task_+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/evaluation/' if parameters["evaluation"] else ''
    os.system(strpath('mkdir '+ parameters["weights_path"] +' '+parameters["write_loss_path"]+' '+parameters["latent_space_path"]+' '+parameters["evaluation_path"]))
    return parameters

def rmbkshn(strings):
    """ Given a string or a list of string remove all final \"\n\" """
    if type(strings) == type(""):
        if strings[-1] == "\n":
            return strings[:-1]
    if type(strings) == type([]):
            new_strings = []
            for i, s in enumerate(strings):
                if s[-1] == "\n":
                    strings[i] = s[:-1]
    return strings
                    
def load_alkane():
    data =  {}
    data["molecule_names"] = []
    data["id"] =  []
    data["boiling_point"] =  []
    data["file_names"] =  []
    with open('data/Alkane/dataset_boiling_point_names.txt') as f:
        for line in f:
            lline = line.split(" ")
            data["id"].append(int(lline[0])-1)
            data["file_names"].append(lline[1])
            data["boiling_point"].append(float(lline[2]))
            data["molecule_names"].append(rmbkshn(lline[3]))
    data["node_positions"] =  []
    data["node_names"] =  []
    data["adjency_matrix"] = [] 
    for i in range(1,151):
        with open('data/Alkane/molecule'+'0'*(3-len(str(i)))+str(i)+'.ct', 'r') as infile:
            lines = infile.readlines()
        n_nodes = int(lines[1].split(" ")[0])
        n_edges = int(rmbkshn(lines[1].split(" ")[1]))
        nodes_lines = rmbkshn(lines[2:2+n_nodes])
        nodes_positions = []
        node_names = []
        for line in  nodes_lines:
            # data["node_names"].append(line[-1])
            node_names.append(line[-1])
            nodes_positions.append( re.split(" * ", line[3:-2] ) )
            if '' in nodes_positions[-1]:
                nodes_positions[-1].remove('')
            nodes_positions[-1] = [ float(ndp) for ndp in nodes_positions[-1] ]
        data["node_names"].append(node_names)
        data["node_positions"].append(np.array(nodes_positions))
        edges_lines = rmbkshn(lines[2+n_nodes:2+n_nodes+n_edges])
        data["adjency_matrix"].append(np.zeros((n_nodes,n_nodes)))
        for line in edges_lines :
            link = [int(n) for  n in re.split(" * ", line )]
            data["adjency_matrix"][-1][link[0]-1,link[1]-1] = link[2]  
        data["adjency_matrix"][-1] = data["adjency_matrix"][-1] + np.transpose(data["adjency_matrix"][-1]) 
    data["graph_size"] = [len(adj) for adj in data["adjency_matrix"]]
    for key in data.keys():
        data[key] = data[key][2:]
    max_degree = np.max([ np.max(np.sum(data["adjency_matrix"][i], axis = 0)) for i in range(len(data["adjency_matrix"])) ])
    data["graph_degree"] = np.array([np.eye(int(max_degree))[np.sum(data["adjency_matrix"][i].astype(np.int32), axis = 0) - 1] for i in range(len(data["adjency_matrix"])) ])
    return data

# def load_drtmnd_dataset(dataset_name, use_attributes_if_exist=True, root_datasets=""):
#     """" Load dataset_name from data folder
#          Any dataset from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
#          can be used without code modification
#          edgels labels starting at value 10 instead of 0
#     """
#     graph_indicator = []
#     graph_size = []
#     graph_labels = []
#     edge_weights = []
#     nb_graph = 0
#     root = os.path.join(root_datasets, dataset_name+'/raw/')
    
#     with open(os.path.join(root,dataset_name+'_graph_indicator.txt'), 'r') as f:
#         lines = f.readlines()
#         graph_indicator = np.array([ int(x.strip()) for x in lines])
#         for i in range(max(graph_indicator)):
#             graph_size.append(graph_indicator[graph_indicator==i+1].shape[0])
#         nb_graph = len(graph_size)

#     with open(os.path.join(root,dataset_name+'_A.txt'), 'r') as f:
#         lines = f.readlines()
#         edge_list = [ [ int(y) for y in x.strip().split(",")] for x in lines]
#         edge_list_bygraph = [[] for x in range(nb_graph)]
#         for i in range(len(edge_list)):
#             index_graph = graph_indicator[edge_list[i][0]-1] - 1
#             edge_list_bygraph[index_graph].append(edge_list[i])
#     # with open('data/'+dataset_name+'/'+dataset_name+'_edge_labels.txt', 'r') as f:
#     #     lines = f.readlines()
#     #     edges_labels = [ int(x.strip())+1 for x in lines] 
#     #     edge_weights = edges_labels
#     with open(os.path.join(root,dataset_name+'_A.txt'), 'r') as f:
#         lines = f.readlines()
#         edges_labels = [ 1 for x in lines] 
#         edge_weights = edges_labels

#     with open(os.path.join(root,dataset_name+'_graph_labels.txt'), 'r') as f:
#         lines = f.readlines()
#         graph_labels = [ int(x.strip()) for x in lines]

#     graph_adjency_matrix = [np.zeros((i,i)) for i in graph_size];
#     p = 0
#     for i in range(len(edge_list_bygraph)):
#         i_ = np.array(edge_list_bygraph[i])
#         i_ = i_ - np.min(i_)
#         for k in i_:
#             graph_adjency_matrix[i][k[0],k[1]] = edge_weights[p]
#             p = p + 1

#     exist_node_labels = False
#     if False: # os.path.exists(os.path.join(root,dataset_name+'_node_labels.txt')) :
#         exist_node_labels = True
#         node_labels = [np.zeros((x,)) for x in graph_size];
#         with open(os.path.join(root,dataset_name+'_node_labels.txt'), 'r') as f:
#             lines = f.readlines()
#             print(lines[0].strip())
#             node_labels_tampon = [ int(x.strip()) for x in lines]
#             # node_labels_tampon = [ 0 for x in lines]

#             k = 0;
#             for i in range(len(graph_size)):
#                 for j in range(graph_size[i]):
#                     node_labels[i][j] = node_labels_tampon[k]
#                     k = k + 1
#         index_max = int(np.max(np.concatenate(np.array(node_labels, dtype = object),axis=0)))
#         graph_node_labels = [ np.eye(index_max+1)[node_labels[i].astype(int)] for i in range(len(graph_adjency_matrix))]

#     exist_node_attributes = False
#     if  os.path.exists(os.path.join(root, dataset_name+'_node_attributes.txt')) :
#         exist_node_attributes = True
#         with open(os.path.join(root,dataset_name+'_node_attributes.txt'), 'r') as f:
#             lines = f.readlines()
#             attributes_size= len(lines[0].replace(" ","").split(','))
#             graph_attributes = [np.zeros((x, attributes_size)) for x in graph_size];
#             k = 0;
#             for i in range(len(graph_size)):
#                 for j in range(graph_size[i]):
#                     graph_attributes [i][j,:] = [float(x) for x in lines[k].replace(" ","").split(',')]
#                     k = k + 1
     
#     data = {}
#     data["graph_size"] = graph_size
#     tampon = {}
#     new_graph_labels = []
#     k = -1
#     for l in graph_labels:
#         if l not in tampon:
#             k = k + 1
#             tampon[l] = k
#         new_graph_labels.append(tampon[l])
#     data["graph_labels"] = np.array(new_graph_labels)
 
#     # data["node_labels"] = np.array(graph_node_labels)
#     if exist_node_attributes and use_attributes_if_exist :
#         if dataset_name == "FRANKENSTEIN" :
#             tampon = np.concatenate( np.array( [ np.concatenate([x,np.zeros((x.shape[0],4))],1) for x in graph_attributes] ) , axis = 0 )
#         else:
#             tampon = np.concatenate( np.array( [ x for x in graph_attributes], dtype=object) , axis = 0 )
#         mean = np.mean(tampon, axis=0)
#         std = np.std(tampon, axis=0)
#         std[std == 0 ] = 1
#         if dataset_name == "FRANKENSTEIN" :
#             graph_attributes = np.array([(np.concatenate([x,np.zeros((x.shape[0],4))],1)-mean)/std for x in graph_attributes])
#         else :
#             graph_attributes = np.array([(x-mean)/std for x in graph_attributes], dtype=object)
#         data["graph_features"] =  np.array(graph_attributes)
#     if exist_node_labels :
#         if dataset_name == "IMDB-BINARY" or  dataset_name == "REDDIT-BINARY" or dataset_name == "IMDB-MULTI":
#             Tamp = [np.sum(adj,0).astype(np.int) for adj in graph_adjency_matrix]
#             max_tamp = max([np.max(t) for t in Tamp])
#             graph_node_labels = [ np.eye(max_tamp)[f-1] for f in Tamp]
#         data["node_labels"] = np.array(graph_node_labels, dtype = object)
#         # print("t")
#     if False: #dataset_name == "ENZYMES" or dataset_name == "PROTEINS_full" or dataset_name == "PROTEINS" or dataset_name == "COX2" or dataset_name == "BZR" or  dataset_name == "Cuneiform":
#         aggregate = [ np.concatenate([ga,gf],axis=1) for ga, gf in zip(graph_attributes,graph_node_labels)]
#         data["graph_fuse"] = np.array(aggregate, dtype=object)
                
#     graph_adj = [(graph_adjency_matrix[i]>0).astype(int) for i in range(len(graph_adjency_matrix))]
#     data["adjency_matrix"] = np.array(graph_adj, dtype = object) 
#     index_to_delete = np.where(np.array(data["graph_size"]) < 2)
#     data["adjency_matrix"] = np.delete(data["adjency_matrix"], index_to_delete, axis = 0 )
#     data["graph_size"] = np.delete(data["graph_size"], index_to_delete, axis = 0)
#     if exist_node_labels :
#         data["node_labels"] = np.delete(data["node_labels"], index_to_delete, axis = 0)
#     if exist_node_attributes :
#         data["graph_features"] = np.delete(data["graph_features"], index_to_delete, axis = 0)
#     if exist_node_attributes and exist_node_labels:
#         data["graph_fuse"] = np.delete(data["graph_fuse"], index_to_delete, axis = 0)
#     data["graph_labels"] = np.delete(data["graph_labels"], index_to_delete, axis = 0)
    

#     max_degree = np.max([ np.max(np.sum(data["adjency_matrix"][i], axis = 0)) for i in range(len(data["adjency_matrix"])) ])
#     # data["features"] = np.array(datam["features"])
#     data["graph_degree"] = np.array([np.eye(int(max_degree))[np.sum(data["adjency_matrix"][i].astype(np.int32), axis = 0) - 1] for i in range(len(data["adjency_matrix"])) ], dtype = object)
    
#     # aggregate = [ np.concatenate([ga,gf],axis=1) for ga, gf in zip(data["graph_degree"],graph_node_labels)]
#     # data["graph_fuse"] = np.array(aggregate)
#     return data

def load_drtmnd_dataset(dataset_name, use_attributes_if_exist=True, root_datasets=""):
    """" Load dataset_name from data folder
         Any dataset from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
         can be used without code modification
         edgels labels starting at value 10 instead of 0
    """
    graph_indicator = []
    graph_size = []
    graph_labels = []
    edge_weights = []
    nb_graph = 0
    root = os.path.join(root_datasets, dataset_name+'/raw/')
    
    with open(os.path.join(root,dataset_name+'_graph_indicator.txt'), 'r') as f:
        lines = f.readlines()
        graph_indicator = np.array([ int(x.strip()) for x in lines])
        for i in range(max(graph_indicator)):
            graph_size.append(graph_indicator[graph_indicator==i+1].shape[0])
        nb_graph = len(graph_size)

    with open(os.path.join(root,dataset_name+'_A.txt'), 'r') as f:
        lines = f.readlines()
        edge_list = [ [ int(y) for y in x.strip().split(",")] for x in lines]
        edge_list_bygraph = [[] for x in range(nb_graph)]
        for i in range(len(edge_list)):
            index_graph = graph_indicator[edge_list[i][0]-1] - 1
            edge_list_bygraph[index_graph].append(edge_list[i])
    # with open('data/'+dataset_name+'/'+dataset_name+'_edge_labels.txt', 'r') as f:
    #     lines = f.readlines()
    #     edges_labels = [ int(x.strip())+1 for x in lines] 
    #     edge_weights = edges_labels
    with open(os.path.join(root,dataset_name+'_A.txt'), 'r') as f:
        lines = f.readlines()
        edges_labels = [ 1 for x in lines] 
        edge_weights = edges_labels

    with open(os.path.join(root,dataset_name+'_graph_labels.txt'), 'r') as f:
        lines = f.readlines()
        graph_labels = [ int(x.strip()) for x in lines]

    graph_adjency_matrix = [np.zeros((i,i)) for i in graph_size];
    p = 0
    for i in range(len(edge_list_bygraph)):
        i_ = np.array(edge_list_bygraph[i])
        i_ = i_ - np.min(i_)
        for k in i_:
            graph_adjency_matrix[i][k[0],k[1]] = edge_weights[p]
            p = p + 1

    exist_node_labels = False
    if os.path.exists(os.path.join(root,dataset_name+'_node_labels.txt')):
        # TODO: RCP changed this...
        exist_node_labels = True
        #node_labels = [np.zeros((x,)) for x in graph_size];
        with open(os.path.join(root,dataset_name+'_node_labels.txt'), 'r') as f:
            # lines = f.readlines()
            # #print(lines[0].strip())
            # print("Line 0 labels:", lines[0], len(lines[0].replace(" ","").split(',')))
            # node_labels_tampon = [ int(x.strip()) for x in lines]
            # # node_labels_tampon = [ 0 for x in lines]

            # k = 0;
            # for i in range(len(graph_size)):
            #     for j in range(graph_size[i]):
            #         node_labels[i][j] = node_labels_tampon[k]
            #         k = k + 1
            lines = f.readlines()
            labels_size= len(lines[0].replace(" ","").split(','))
            #print(f"Line 0 labels ({len(lines[0].replace(' ','').split(','))}):", lines[0])
            
            node_labels = [np.zeros((x, labels_size)) for x in graph_size];
            k = 0;
            for i in range(len(graph_size)):
                for j in range(graph_size[i]):
                    node_labels[i][j,:] = [float(x) for x in lines[k].replace(" ","").split(',')]
                    k = k + 1
        #print("NN0: ", np.max([np.max(u) for u in node_labels]))
        #print("NL", [set(u) for u in node_labels])
        #index_max = int(np.max(np.concatenate(np.array(node_labels, dtype = object),axis=0)))
        #graph_node_labels = [ np.eye(index_max+1)[node_labels[i].astype(int)] for i in range(len(graph_adjency_matrix))]
        #print("NL2", graph_node_labels[0].shape, graph_node_labels)
        graph_node_labels = node_labels

    exist_node_attributes = False
    if  os.path.exists(os.path.join(root, dataset_name+'_node_attributes.txt')) :
        exist_node_attributes = True
        with open(os.path.join(root,dataset_name+'_node_attributes.txt'), 'r') as f:
            lines = f.readlines()
            
            attributes_size= len(lines[0].replace(" ","").split(','))
            
            #print(f"Line 0 attr ({len(lines[0].replace(' ','').split(','))}):", lines[0])
            
            graph_attributes = [np.zeros((x, attributes_size)) for x in graph_size];
            k = 0;
            for i in range(len(graph_size)):
                for j in range(graph_size[i]):
                    graph_attributes [i][j,:] = [float(x) for x in lines[k].replace(" ","").split(',')]
                    k = k + 1
     
    data = {}
    data["graph_size"] = graph_size
    tampon = {}
    new_graph_labels = []
    k = -1
    for l in graph_labels:
        if l not in tampon:
            k = k + 1
            tampon[l] = k
        new_graph_labels.append(tampon[l])
    data["graph_labels"] = np.array(new_graph_labels)
 
    # data["node_labels"] = np.array(graph_node_labels)
    if exist_node_attributes and use_attributes_if_exist :
        if dataset_name == "FRANKENSTEIN" :
            tampon = np.concatenate( np.array( [ np.concatenate([x,np.zeros((x.shape[0],4))],1) for x in graph_attributes] ) , axis = 0 )
        else:
            tampon = np.concatenate( np.array( [ x for x in graph_attributes], dtype=object) , axis = 0 )
        mean = np.mean(tampon, axis=0)
        std = np.std(tampon, axis=0)
        std[std == 0 ] = 1
        if dataset_name == "FRANKENSTEIN" :
            graph_attributes = np.array([(np.concatenate([x,np.zeros((x.shape[0],4))],1)-mean)/std for x in graph_attributes])
        else :
            graph_attributes = np.array([(x-mean)/std for x in graph_attributes], dtype=object)
        data["graph_features"] =  np.array(graph_attributes)
    if exist_node_labels :
        if dataset_name == "IMDB-BINARY" or  dataset_name == "REDDIT-BINARY" or dataset_name == "IMDB-MULTI":
            Tamp = [np.sum(adj,0).astype(np.int) for adj in graph_adjency_matrix]
            max_tamp = max([np.max(t) for t in Tamp])
            graph_node_labels = [ np.eye(max_tamp)[f-1] for f in Tamp]
        data["node_labels"] = np.array(graph_node_labels, dtype = object)
        # print("t")
    if dataset_name == "ENZYMES" or dataset_name == "PROTEINS_full" or dataset_name == "PROTEINS" or dataset_name == "COX2" or dataset_name == "BZR" or  dataset_name == "Cuneiform":
        aggregate = [ np.concatenate([ga,gf],axis=1) for ga, gf in zip(graph_attributes,graph_node_labels)]
        data["graph_fuse"] = np.array(aggregate, dtype=object)
                
    graph_adj = [(graph_adjency_matrix[i]>0).astype(int) for i in range(len(graph_adjency_matrix))]
    data["adjency_matrix"] = np.array(graph_adj, dtype = object) 
    index_to_delete = np.where(np.array(data["graph_size"]) < 2)
    data["adjency_matrix"] = np.delete(data["adjency_matrix"], index_to_delete, axis = 0 )
    data["graph_size"] = np.delete(data["graph_size"], index_to_delete, axis = 0)
    if exist_node_labels :
        data["node_labels"] = np.delete(data["node_labels"], index_to_delete, axis = 0)
    if exist_node_attributes :
        data["graph_features"] = np.delete(data["graph_features"], index_to_delete, axis = 0)
    if exist_node_attributes and exist_node_labels:
        data["graph_fuse"] = np.delete(data["graph_fuse"], index_to_delete, axis = 0)
    data["graph_labels"] = np.delete(data["graph_labels"], index_to_delete, axis = 0) 

    max_degree = np.max([ np.max(np.sum(data["adjency_matrix"][i], axis = 0)) for i in range(len(data["adjency_matrix"])) ])
    # data["features"] = np.array(datam["features"])
    data["graph_degree"] = np.array([np.eye(int(max_degree))[np.sum(data["adjency_matrix"][i].astype(np.int32), axis = 0) - 1] for i in range(len(data["adjency_matrix"])) ], dtype = object)
    
    # aggregate = [ np.concatenate([ga,gf],axis=1) for ga, gf in zip(data["graph_degree"],graph_node_labels)]
    # data["graph_fuse"] = np.array(aggregate)
    return data

############## WWL utils function
# Next function come from https://github.com/BorgwardtLab/WWL
# Implement features extraction of WWL papers 
def create_labels_seq_cont(node_features, adj_mat, h):
	'''
	    create label sequence for continuously attributed graphs 
	'''
	n_graphs = len(node_features)
	labels_sequence = []
	for i in range(n_graphs):
		graph_feat = []

		for it in range(h+1):
			if it == 0:
				graph_feat.append(node_features[i])
			else:
				adj_cur = adj_mat[i]+np.identity(adj_mat[i].shape[0])
				adj_cur = create_adj_avg(adj_cur)

				np.fill_diagonal(adj_cur, 0)
				graph_feat_cur = 0.5*(np.dot(adj_cur, graph_feat[it-1]) + graph_feat[it-1])
				graph_feat.append(graph_feat_cur)

		labels_sequence.append(np.concatenate(graph_feat, axis = 1))
		if i % 100 == 0:
			print(f'Processed {i} graphs out of {n_graphs}')
	
	return labels_sequence

def create_adj_avg(adj_cur):
	'''
	    create adjacency
	'''
	deg = np.sum(adj_cur, axis = 1)
	deg = np.asarray(deg).reshape(-1)

	deg[deg!=1] -= 1

	deg = 1/deg
	deg_mat = np.diag(deg)
	adj_cur = adj_cur.dot(deg_mat.T).T
	
	return adj_cur

def compute_wl_embeddings_discrete(node_features, adj_mat, h):
    graphs = [ ig.Graph.Adjacency((adj > 0).tolist()) for adj in adj_mat ]
    for g, f in zip(graphs, node_features):
        for i in range(len(g.vs)):
            g.vs[i]["label"]  = str(f[i].argmax(0))
    # graph_filenames = retrieve_graph_filenames(data_directory)
    # graphs = [ig.read(filename) for filename in graph_filenames]
    wl = WeisfeilerLehman()
    label_dicts = wl.fit_transform(graphs, h)
    # Each entry in the list represents the label sequence of a single
    # graph. The label sequence contains the vertices in its rows, and
    # the individual iterations in its columns.
    #
    # Hence, (i, j) will contain the label of vertex i at iteration j.
    label_sequences = [
        np.full((len(graph.vs), h + 1), np.nan) for graph in graphs
    ]   
    for iteration in sorted(label_dicts.keys()):
        for graph_index, graph in enumerate(graphs):
            labels_raw, labels_compressed = label_dicts[iteration][graph_index]
            # Store label sequence of the current iteration, i.e. *all*
            # of the compressed labels.
            label_sequences[graph_index][:, iteration] = labels_compressed
    return label_sequences

class WeisfeilerLehman(TransformerMixin):
    """ Class that implements the Weisfeiler-Lehman transform
        Credits: Christian Bock and Bastian Rieck
    """
    def __init__(self):
        self._relabel_steps = defaultdict(dict)
        self._label_dict = {}
        self._last_new_label = -1
        self._preprocess_relabel_dict = {}
        self._results = defaultdict(dict)
        self._label_dicts = {}

    def _reset_label_generation(self):
        self._last_new_label = -1

    def _get_next_label(self):
        self._last_new_label += 1
        return self._last_new_label

    def _relabel_graphs(self, X: List[ig.Graph]):
        num_unique_labels = 0
        preprocessed_graphs = []
        for i, g in enumerate(X):
            x = g.copy()
            
            if not 'label' in x.vs.attribute_names():
                x.vs['label'] = list(map(str, [l for l in x.vs.degree()]))           
            labels = x.vs['label']
            

            new_labels = []
            for label in labels:
                if label in self._preprocess_relabel_dict.keys():
                    new_labels.append(self._preprocess_relabel_dict[label])
                else:
                    self._preprocess_relabel_dict[label] = self._get_next_label()
                    new_labels.append(self._preprocess_relabel_dict[label])
            x.vs['label'] = new_labels
            self._results[0][i] = (labels, new_labels)
            preprocessed_graphs.append(x)
        self._reset_label_generation()
        return preprocessed_graphs

    def fit_transform(self, X: List[ig.Graph], num_iterations: int=3):
        X = self._relabel_graphs(X)
        for it in np.arange(1, num_iterations+1, 1):
            self._reset_label_generation()
            self._label_dict = {}
            for i, g in enumerate(X):
                # Get labels of current interation
                current_labels = g.vs['label']

                # Get for each vertex the labels of its neighbors
                neighbor_labels = self._get_neighbor_labels(g, sort=True)

                # Prepend the vertex label to the list of labels of its neighbors
                merged_labels = [[b]+a for a,b in zip(neighbor_labels, current_labels)]

                # Generate a label dictionary based on the merged labels
                self._append_label_dict(merged_labels)

                # Relabel the graph
                new_labels = self._relabel_graph(g, merged_labels)
                self._relabel_steps[i][it] = { idx: {old_label: new_labels[idx]} for idx, old_label in enumerate(current_labels) }
                g.vs['label'] = new_labels

                self._results[it][i] = (merged_labels, new_labels)
            self._label_dicts[it] = copy.deepcopy(self._label_dict)
        return self._results

    def _relabel_graph(self, X: ig.Graph, merged_labels: list):
        new_labels = []
        for merged in merged_labels:
            new_labels.append(self._label_dict['-'.join(map(str,merged))])
        return new_labels

    def _append_label_dict(self, merged_labels: List[list]):
        for merged_label in merged_labels:
            dict_key = '-'.join(map(str,merged_label))
            if dict_key not in self._label_dict.keys():
                self._label_dict[ dict_key ] = self._get_next_label()

    def _get_neighbor_labels(self, X: ig.Graph, sort: bool=True):
            neighbor_indices = [[n_v.index for n_v in X.vs[X.neighbors(v.index)]] for v in X.vs]
            neighbor_labels = []
            for n_indices in neighbor_indices:
                if sort:
                    neighbor_labels.append( sorted(X.vs[n_indices]['label']) )
                else:
                    neighbor_labels.append( X.vs[n_indices]['label'] )
            return neighbor_labels

############## GIN utils function
class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

    def __len__(self):
        return len(self.node_tags)

def process_data_gin(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    triggered = False
    for i, data in enumerate(dataset["graph_adjency_matrix"]):
        g=nx.from_numpy_matrix(data.astype(np.int))
        # node_tags = [int(x) for x in dataset["node_labels"][i]] #Quand les labels = entier
        node_tags = [int(x.argmax(0)) for x in dataset["node_labels"][i]]
        if dataset["graph_labels"][i] not in label_dict:
            mapped = len(label_dict)
            label_dict[dataset["graph_labels"][i]] = mapped
        g_list.append(S2VGraph(g, dataset["graph_labels"][i], node_tags))

            
    #This is the 2nd part of the load_data code.
    #Within each graph class, he is creating the following:
    #1. graph.neighbors <matrix> : This is similar to the matrix A, but it only contains the data of the connections.
    #2. graph.max_neighbor <int>: This is basically the maximum number of connections that arises from a single node in the graph
    #3. graph.label <int>        : This is to ensure that the labels do not skip numbers, i.e. labels <1,2,4,5> becomes <0,1,2,3>
    #4. graph.edge_mat <matrix>  : This is a transpose of the edge matrix from the graph [[nodes][connected nodes]]
            
    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
            
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]  #This line appears to be redundant
            degree_list.append(len(g.neighbors[i]))         
        g.max_neighbor = max(degree_list)
        g.label = label_dict[g.label]  #All this line does, is ensure that there is no skipping of labels. The actual numerical label is just a variable.
    
        #These two lines create an extension of the edges to ensure that they are bi-directional
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        
        g.edge_mat = tf.transpose(tf.constant(edges))
        
        
        
    #This part gives the degree dictionary but not in the order of nodes within the dataset.txt file  
    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))
    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}
    
    k = -1
    for g in g_list:
        k = k + 1
        #Quand les labels = entier
        # node_features = np.zeros((len(g.node_tags), len(tagset)))
        # node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1  
        # g.node_features = tf.constant(node_features)
        g.node_features = tf.constant(dataset["node_labels"][k])


        
        
    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return np.array(g_list)#, len(label_dict)
