import pandas as pd

import torch
from torch import nn
from torch import Tensor
import math
import configparser
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import cuda
import budformer as bf
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
import dba
import tslearn
import tslearn.barycenters as bc
import tslearn.metrics
import numpy as np
import tqdm
from scipy.cluster import hierarchy
import pickle
import time
import random



def ts_linkage(array: np.array, method='closest', metric=tslearn.metrics.dtw):
    print("Constructing distance matrix")
    n = array.shape[0]
    linkage_matrix = np.zeros((n-1,4))
    distance_matrix = np.ones((2*n,2*n)) * np.Inf


    ctr = 0

    for i in tqdm.tqdm(range(n)):
        for j in range(i+1,n):
            distance_matrix[i,j] = metric(array[i],array[j])
            ctr += 1
    #distance_matrix += np.transpose(distance_matrix)

    print(ctr)

    orig_distance_matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if i==j:
                orig_distance_matrix[i,j] = 0
            else:
                orig_distance_matrix[i,j] = min(distance_matrix[i,j],distance_matrix[j,i])

    cluster_composition = []
    removed_clusters = []
    cluster_centres = array.tolist()

    for i in range(n):
        cluster_composition.append([i])

    for i in tqdm.tqdm(range(n-1)): #we have n iterations to do
        #check for the smallest distance
        min_val = np.min(distance_matrix)
        min_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        
        if np.isinf(min_val):
            #check if this is the last cluster:
            if len(cluster_centres) == 2*n - 2:
                remaining_indices = list(set(range(2*n-2)) - set(removed_clusters))
                min_idx = np.array(remaining_indices)
                min_val = metric(cluster_centres[int(min_idx[0])], cluster_centres[int(min_idx[1])])
        
        if int(min_idx[0]) in removed_clusters or int(min_idx[1]) in removed_clusters:
            print("Error")


        linkage_matrix[i,0] = int(min_idx[0])
        linkage_matrix[i,1] = int(min_idx[1])
        linkage_matrix[i,2] = min_val
        cluster_composition.append(cluster_composition[min_idx[0]]+cluster_composition[min_idx[1]])

        #print(cluster_composition[-1])

        linkage_matrix[i,3] = len(cluster_composition[-1])

        #remove points that were clustered
        distance_matrix[min_idx[0]] = np.ones((1,2*n))*np.Inf
        distance_matrix[min_idx[1]] = np.ones((1,2*n))*np.Inf

        distance_matrix[:,min_idx[0]] = np.ones((1,2*n))*np.Inf
        distance_matrix[:,min_idx[1]] = np.ones((1,2*n))*np.Inf

        removed_clusters.append(int(min_idx[0]))
        removed_clusters.append(int(min_idx[1]))

        #calculate the center of the cluster using DBA

        #dissect into dimensions, perform DBA, return

        #we use only 2 centers and weigh them

        centres_to_average = np.array([cluster_centres[min_idx[0]],cluster_centres[min_idx[1]]])
        weights_to_average = np.array([len(cluster_centres[min_idx[0]]),len(cluster_centres[min_idx[1]])])

  
        cluster_centres.append(bc.dtw_barycenter_averaging(centres_to_average,weights=weights_to_average))

        for j in range(n+i-1):

            #calculate distance between j and the new center
            if j not in removed_clusters:
                distance = metric(cluster_centres[j], cluster_centres[-1])
                if np.isinf(distance):
                    print("Error distance is inf")
                distance_matrix[n+i,j] = distance

    return orig_distance_matrix, linkage_matrix, cluster_centres


def run(config,set,output,maxclusters,dump_location,drop_id = True, split_date = False, window_length=720,prediction_window=30, max_clustering_samples = 1000):

    train_dataset, test_dataset = bf.CustomDataSet(set,window_length=window_length,prediction_window=prediction_window,require_date_split=split_date,drop_idx_column=drop_id).getSets()
    #if you want to test flow shorten windows

    #visualize_index = 0
    #input_list = train_dataset[0][0][:,visualize_index].tolist()

    #plt.plot(input_list)
    #plt.show()

    #split into test/train:

    datalist = []

    #only choose a limited number of samples

    bucket_distances = float(len(train_dataset)) / max_clustering_samples
    pivot_locations = []
    bucket_range = 0

    for _ in range(max_clustering_samples):

        pivot = int(bucket_range)
        # randomly mutate
        offset = random.randint(-1*int(bucket_distances),int(bucket_distances))
        pivot += offset

        if pivot < 0:
            pivot = 0
        if pivot >= len(train_dataset):
            pivot = len(train_dataset)-1

        pivot_locations.append(pivot)
        bucket_range += bucket_distances


    for i, e in enumerate(train_dataset):
        if i in pivot_locations:
            datalist.append(e[0])

    datalist_array = np.array(datalist)

    distances, linkage, centres = ts_linkage(datalist_array) #so it computes faster


    print(linkage)

    dn = hierarchy.dendrogram(linkage)

    linkage.shape

    results = [-1, -1, -1]

    for i in range(4,maxclusters):
        clust = hierarchy.fcluster(linkage,i,criterion='maxclust')
        results.append(float(silhouette_score(distances,clust,metric="precomputed")))

    val, idx = max((val, idx) for (idx, val) in enumerate(results))

    print(val)
    print(idx)

    best_cluster_number = idx

    #file = open(dump_location,"wb")

    #pickle.dump(linkage,file)

    #file.close()

    params = bf.ParameterProvider(config)

    transformer = bf.BudFormer(linkage_matrix=linkage,cluster_centers=centres,config=params)

    

    transformer.form(best_cluster_number)

    no_params = sum(p.numel() for p in transformer.parameters())
    print("Params: ", no_params)

    transformer.train_cuda(train_dataset,torch.cuda.current_device(),epochs=params.provide("epochs"))

    print("Validation result ",transformer.validate_cuda(test_dataset,torch.cuda.current_device()))
    transformer.output_and_save_only_prediction(test_dataset[0][0],test_dataset[0][2],torch.cuda.current_device(),2,dump_location,output)

    inp, out, pred = transformer.return_numbers(test_dataset[0][0],test_dataset[0][2],torch.cuda.current_device(),-5,dump_location,output)

    

    return inp, out, pred