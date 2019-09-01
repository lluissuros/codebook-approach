
'''
this could be a playground for interpretability and so on.
'''

import os
import json
import warnings
import numpy as np
import pickle

from settings import EMBEDDING_FOLDERS

def load_as_binary(filename = './files/no_name.file'):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def save_as_binary(element, filename = './files/no_name.file'):
    print('saving as pickle binary ...')
    with open(filename, "wb") as f:
        pickle.dump(element, f, pickle.HIGHEST_PROTOCOL)
    print('... file saved as', filename)


def compute_metrics_difference(dataset1, dataset2, embedding_name):
    if(dataset1 == None or dataset2 == None):
        return None
    ds1_metrics = dataset1['evaluation_metrics_{}'.format(embedding_name)]
    ds2_metrics = dataset2['evaluation_metrics_{}'.format(embedding_name)] 
    metrics_difference = {}
    metrics_difference['purity'] = ds1_metrics['purity'] - ds2_metrics['purity']
    metrics_difference['adjusted_mutual_info'] = ds1_metrics['adjusted_mutual_info'] - ds2_metrics['adjusted_mutual_info']
    metrics_difference['adjusted_rand'] = ds1_metrics['adjusted_rand'] - ds2_metrics['adjusted_rand']
    return metrics_difference

def make_datasets_report(metrics_differences, fieldname, ds_name_1, ds_name_2):
    ds1_wins = []
    ds2_wins = []
    discarded = []
    for idx, metric_diff in enumerate(metrics_differences):
        if metric_diff == None: discarded.append(idx)
        elif(metric_diff[fieldname]>=0): 
            ds1_wins.append(idx)
        else: ds2_wins.append(idx)
    
    usable_datasets = len(metrics_differences) - len(discarded)

    print('\n*',fieldname.upper(), ":\n ")
    print(" -", ds_name_1, "had better performance in: ")
    print(ds1_wins)
    print(len(ds1_wins) ,"/", usable_datasets, "\n")
    print(" -", ds_name_2, "had better performance in ")
    print(ds2_wins)
    print(len(ds2_wins) ,"/", usable_datasets, "\n")
    print(" -", len(discarded), "discarded datasets :") 
    print(discarded, "\n")
    print("numerical difference:", metrics_differences)
    


def fix_naming(dataset):
    '''TODO: remove when I dont use it anymore'''
    try:
        dataset['evaluation_metrics_audioset'] = dataset['evaluation_metrics_codebook_audioset']
    except:
        dataset = None 
    return dataset


def compare_datasets(datasets1, datasets2, ds_name_1="ds1", ds_name_2="ds2" ):
    #combined_datasets = [{**ds1, **ds2} for ds1, ds2 in zip(datasets_mean, datasets_codebook)]
    for embedding_name, _ in EMBEDDING_FOLDERS.items():
        metrics_differences = [compute_metrics_difference(ds1, ds2, embedding_name) for ds1, ds2 in zip(datasets1, datasets2)]
        make_datasets_report(metrics_differences, 'purity', ds_name_1, ds_name_2)
        make_datasets_report(metrics_differences, 'adjusted_mutual_info', ds_name_1, ds_name_2)
        make_datasets_report(metrics_differences, 'adjusted_rand', ds_name_1, ds_name_2)
            


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        #for codebook_size in [64, 128, 256, 512, 1024, 2048]:
        for codebook_size in [64, 128, 256, 512]:
            print("\n================================")
            print("\nCodebook size: ", codebook_size)

            #load
            filename_codebook_loc =  "files_audioSet_julio/"+str(codebook_size)+"_codebook_clusters.file" 
            #filename_codebook_glob=  "files/"+str(codebook_size)+"_global_codebook_clusters.file" 
            filename_mean =  "files/mean_computed_datasets_clusters.file" #TODO: change name to mean
            datasets_codebook_loc = load_as_binary(filename_codebook_loc)
            #datasets_codebook_glob = load_as_binary(filename_codebook_glob)
            datasets_mean = load_as_binary(filename_mean)

            #compare_datasets(datasets_codebook_glob, datasets_codebook_loc, "global codebook", "local codebook")
            #compare_datasets(datasets_codebook_glob, datasets_mean, "global codebook", "original mean dataset")
            compare_datasets(datasets_codebook_loc, datasets_mean, "local codebook", "original mean dataset")
