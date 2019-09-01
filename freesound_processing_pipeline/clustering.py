import os
import sys
import json
import time
import operator
import warnings
import numpy as np
import networkx as nx
import pickle
from networkx.readwrite import json_graph
import community.community_louvain as com
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer #maybe not used?
from sklearn.feature_extraction.text import TfidfVectorizer




from settings import EMBEDDING_FOLDERS


def remove_failed_embeddings(datasets):
    """
    Check all embedding folders to list missing files.
    Remove them from the datasets.

    """
    all_ids = json.load(open('all_sound_ids.json', 'rb'))
    missing_embeddings = []

    # parse all embedding folders and see which files are missing
    for embedding_folder in EMBEDDING_FOLDERS.values():
        if embedding_folder:
            embeddings_files = os.listdir(embedding_folder)
            for s_id in all_ids:
                if '{0}.npy'.format(s_id) not in embeddings_files:
                    missing_embeddings.append(s_id)

    missing_embeddings = set(missing_embeddings)

    # remove sounds in the datasets
    for dataset in datasets:
        for s_id in missing_embeddings:
            try:
                dataset['sound_ids'].remove(s_id)
            except:
                pass
        for _, obj in dataset['dataset'].items():
            for s_id in missing_embeddings:
                try:
                    obj.remove(s_id)
                except:
                    pass


def create_label_vector(dataset, ontology_by_id):
    """
    Returns given dataset with some label fields.

    """
    labels = []
    label_ids = []
    label_names = []
    sound_ids = []
    for node_id, obj in dataset.items():
        label_ids.append(node_id)
        label_names.append(ontology_by_id[node_id]['name'])
        sound_ids += obj
        for sound_id in obj:
            labels.append(label_ids.index(node_id))
    return sound_ids, labels, label_ids, label_names


def load_features(sound_ids, embedding_folder):
    """
    Returns given dataset with embedding features.

    """
    if embedding_folder:
        embedding_files = [embedding_folder + '{0}.npy'.format(sound_id)
                        for sound_id in sound_ids]

        features = [np.load(f) for f in embedding_files]
    return features


def statistical_aggregation_features(features):
    '''
    At the moment, we perform only mean
    '''
    X_mean = [np.mean(f, axis=0) for f in features]
    return X_mean




def create_merged_features(dataset):
    """
    Merges Audioset and OpenL3 features.
    Adds a field to EMBEDDING_FOLDERS dict.
    WARNING: must be called after load_features().
    TODO: now it should use the X_mean values
    """
    dataset['X_audioset_openl3'] = [np.concatenate((a,b)) for a, b in 
                                    zip(dataset['X_audioset'], dataset['X_openl3-music'])]
    dataset['X_audioset_openl3'] = preprocessing.scale(dataset['X_audioset_openl3'])
    # dataset['X_audioset_openl3'] = PCA(n_components=100).fit_transform(dataset['X_audioset_openl3'])
    EMBEDDING_FOLDERS['audioset_openl3'] = None

    return dataset


def clean(dataset):
    """
    TODO: maybe add a clean for empty features or None.
    OLD CODE FROM OLD PROJECT

    """
    sound_idx_to_remove = set()
    for idx, feature in enumerate(dataset['X_AS']):
        if not feature.any():
            sound_idx_to_remove.add(idx)
    for idx, feature in enumerate(dataset['X_FS']):
        if not feature:
            sound_idx_to_remove.add(idx)
    sound_idx_to_remove = list(sound_idx_to_remove)
    dataset['labels'] = np.delete(dataset['labels'], sound_idx_to_remove)
    dataset['X_AS'] = np.delete(dataset['X_AS'], sound_idx_to_remove, 0)
    dataset['X_FS'] = [obj for idx, obj 
                       in enumerate(dataset['X_FS']) 
                       if idx not in sound_idx_to_remove]
    dataset['sound_ids'] = np.delete(dataset['sound_ids'], sound_idx_to_remove)
    return dataset


def scale_features(dataset):
    """
    TODO: maybe scale features in a dataset.
    OLD CODE FROM OLD PROJECT

    """
    dataset['X_AS_p'] = preprocessing.scale(dataset['X_AS'])
    dataset['X_FS_p'] = preprocessing.scale(dataset['X_FS'])
    dataset['X_FS_p'] = PCA(n_components=100).fit_transform(dataset['X_FS_p'])
    return dataset


def compute_similarity_matrix(X):
    """
    Compute similarity matrix of the given features.

    """
    euclidian_distances = euclidean_distances(X)
    similarity_matrix = 1 - euclidian_distances/euclidian_distances.max()
    similarity_matrix = np.exp(-1 * euclidian_distances / euclidian_distances.std())
    # similarity_matrix = cosine_similarity(X)
    return similarity_matrix


def log_clustering(name, eval_metrics, message=""):
    '''print on the terminal what we find interesting'''
    print('\n================================================')
    print('\n{} dataset:'.format(name))
    print(message)
    print('\n')
    print(eval_metrics)





def cluster_dataset(true_labels, similarity_matrix):
    """
    TODO: Now this is a separate function because of design pattern issues.
    It should only be one single function cluster_dataset(X, labels, similarity_matrix)
    I do it now for test, but when I take EMBEDDING_FOLDERS.items() loop out, it will be possible and nicer
    """
    labels, graph_json = cluster(similarity_matrix)
    purity, adjusted_mutual_info, adjusted_rand = evaluate(labels, true_labels)
    
    eval_metrics = {
        'purity': purity,
        'adjusted_mutual_info': adjusted_mutual_info,
        'adjusted_rand': adjusted_rand        
    }

    return labels, eval_metrics, graph_json


def cluster(similarity_matrix):
    """
    Apply clustering with the features given as input.

    """
    labels_knn, graph_json = knn_graph_clustering(similarity_matrix, 8)
    return labels_knn, graph_json


def knn_graph_clustering(similarity_matrix, k):
    """
    Apply k-nn graph-based clustering on items of the given similarity matrix.
    
    """
    graph = create_knn_graph(similarity_matrix, k)
    classes = com.best_partition(graph)

    # export clustered graph as json
    nx.set_node_attributes(graph, classes, 'group')
    graph_json = json_graph.node_link_data(graph)

    return [classes[k] for k in range(len(classes.keys()))], graph_json


def create_knn_graph(similarity_matrix, k):
    """ 
    Returns a k-nn graph from a similarity matrix - NetworkX module.

    """
    np.fill_diagonal(similarity_matrix, 0) # for removing the 1 from diagonal
    g = nx.Graph()
    g.add_nodes_from(range(len(similarity_matrix)))
    for idx in range(len(similarity_matrix)):
        g.add_edges_from([(idx, i) for i in nearest_neighbors(similarity_matrix, idx, k)])
    return g  

    
def nearest_neighbors(similarity_matrix, idx, k):
    """
    Returns the k nearest meighbots in the similarity matrix.
    """
    distances = []
    for x in range(len(similarity_matrix)):
        distances.append((x,similarity_matrix[idx][x]))
    distances.sort(key=operator.itemgetter(1), reverse=True)
    return [d[0] for d in distances[0:k]]    


def evaluate(predicted_labels, true_labels):
    """
    Returns different metrics of the evaluation of the predicted labels against the true labels.

    """
    adjusted_rand = metrics.adjusted_rand_score(true_labels, predicted_labels)
    adjusted_mutual_info = metrics.adjusted_mutual_info_score(true_labels, predicted_labels)
    purity = purity_score(np.array(true_labels), np.array(predicted_labels))
    return round(purity, 4), round(adjusted_mutual_info, 4), round(adjusted_rand,4)


def purity_score(y_true, y_pred):
    """
    Returns the purity score.
    
    """
    # matrix which will hold the majority-voted labels
    y_labeled_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_labeled_voted[y_pred==cluster] = winner
    return metrics.accuracy_score(y_true, y_labeled_voted)



def save_graph_json_for_web_visu(graph_json, sound_ids, ds_name, embedding_name):
    # save clustered graph, with metadata, as json file for web app
    metadata = json.load(open('json/sounds_metadata.json', 'rb'))

    for node in graph_json['nodes']:
        node.update({ 'sound_id': sound_ids[node['id']] })
        try:
            node.update(metadata[str(node['sound_id'])])
        except:
            pass

    filename = 'web-visu/json/{}-{}.json'.format(ds_name, embedding_name)
    #filename = 'web-visu/json/{}.json'.format(ds_name) # like this for new web-visu? 
    json.dump(graph_json, open(filename, 'w'))



def add_clustering_info_for_web_visu(datasets):
    previous_info = load_json('web-visu/clustering_info.json')
    clustering_info = {
        'datasets': [dataset['web_visu_dataset_name'] for dataset in datasets],
        'features': [embedding_name for embedding_name, _ in EMBEDDING_FOLDERS.items()]
    }

    if len(previous_info) > 0 :
        updated_info = {
            'datasets': previous_info['datasets'] + clustering_info['datasets'],
            'features': [embedding_name for embedding_name, _ in EMBEDDING_FOLDERS.items()],
        }
    else: updated_info = clustering_info

    json.dump(updated_info, open('web-visu/clustering_info.json', 'w'))


def save_to_json(element, filename = './files/no_name.json'):
    print('saving to json ...')
    with open(filename, 'w') as fp:
        json.dump(element, fp, sort_keys=True, indent=4) #pretty json
    print('... file saved as', filename)


def load_json(filename = './files/no_name.json'):
    with open(filename) as json_file:  
        data = json.load(json_file)
    return data

def save_as_binary(element, filename = './files/no_name.file'):
    print('saving as pickle binary ...')
    with open(filename, "wb") as f:
        pickle.dump(element, f, pickle.HIGHEST_PROTOCOL)
    print('... file saved as', filename)

def load_as_binary(element, filename = './files/no_name.file'):
    #TODO: remove element argument
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def get_all_features_as_list(features):
    all_features = []
    for embedded_sound in features:
        for embeddings in embedded_sound:
            all_features.append(embeddings)
    return all_features


# CODEBOOk FUNCS
def generate_codebooks(features, n_words=128):
    '''
    Will create a codebook (cluster centroids given all frames in the dataset)
    TODO: 
    * Try Minibatch k-means, faster but more variance on final results
    * Is it meaningful to have a big cluster number (eg 1024) for a small dataset (eg: 4000)?
    '''
    all_features = get_all_features_as_list(features)
    X = np.array(all_features)

    k_means = KMeans(
        n_clusters=n_words) #cluster  

    start_time = time.time()
    k_means.fit_predict(X)
    elapsed_time = time.time() - start_time 
    
    print(n_words, 'n_words length codebook took ', elapsed_time, 'seconds to compute')

    labels = k_means.labels_ 
    centroids = k_means.cluster_centers_ #take the cluster center
    codebook =  {
        'centroids': centroids, 
        'labels': labels,
        'n_words': n_words}
    return codebook



def encode_frames(features, codebook):
    '''
    Encode each frame against the codebook:
    maps each sample to the nearest centroid
    '''
    encoded_audios = []
    centroids = codebook['centroids']
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centroids)
    for embedded_sound in features:
        _, centroid_idxs = nbrs.kneighbors(embedded_sound) #obtain the nearest centroid index for each frame
        encoded_audios.append(centroid_idxs.squeeze())  # squeeze will get rid of unnecesary dimesions
    return encoded_audios


   
def create_histograms(codebook, encoded_audios):
    '''
    returns a dictionary with id and the corresponding histogram of encoded frames.
    An histogram is a representation of the value distribution.
    '''
    centroids = codebook['centroids']
    
    histograms = []
    bins = np.arange(0, len(centroids) + 1) - 0.5 # bins centered
    for encoded_track in encoded_audios:
        histogram, _ = np.histogram(encoded_track, bins)
        histograms.append(histogram)
    return histograms



def compute_similarity_matrix_from_tfidf(encoded_tracks):
    '''
    Performs Tf-IDF over the encoded tracks corpus
    TODO: check if the corpus should be ALL encoded tracks, not only local dataset. I think it makes sense, normally tfidf is over a big corpus
    '''
    def encode_track_as_text(track): 
        '''returns a string. We prefix cluster indexs with a character, otherwise tfidf ignore plain numbers'''
        if (track.size < 2):
            return 'c' + str(track)
        else:    
            return ' '.join('c' + str(c) for c in track)
    
    vectorizer = TfidfVectorizer(preprocessor = encode_track_as_text) 
    docs_tfidf = vectorizer.fit_transform(encoded_tracks)
    similarity_matrix = cosine_similarity(docs_tfidf)
    return similarity_matrix
    

def get_all_frames(datasets, ontology_by_id):
    all_frames = []
    for _, embedding_folder in EMBEDDING_FOLDERS.items():
        for dataset in datasets:
            sound_ids, _, _, _ = create_label_vector(dataset['dataset'], ontology_by_id)
            features = load_features(sound_ids, embedding_folder)
            all_frames += features
    return all_frames  


def have_enough_data_to_cluster(features, codebook_size):
    '''check if we have enough data, heuristcally'''
    all_data_points = len(get_all_features_as_list(features))
    min_datapoints_per_cluster = 10 #aproximation
    return all_data_points > (codebook_size * min_datapoints_per_cluster)



def global_codebook_encoding_main(codebook_sizes, datasets, ontology_by_id):
    for codebook_size in codebook_sizes:
        all_frames = get_all_frames(datasets, ontology_by_id)
        codebook_global = generate_codebooks(all_frames, codebook_size) #generate global codebook
        for dataset in datasets:
            sound_ids, true_labels, label_ids, label_names = create_label_vector(dataset['dataset'], ontology_by_id)
            
            for embedding_name, embedding_folder in EMBEDDING_FOLDERS.items():
                features = load_features(sound_ids, embedding_folder)

                encoded_audios = encode_frames(features, codebook_global)
                histograms = create_histograms(codebook_global, encoded_audios) # maybe not really needed, only for plotting
                
                similarity_matrix = compute_similarity_matrix_from_tfidf(encoded_audios)
                cluster_labels, eval_metrics, graph = cluster_dataset(true_labels, similarity_matrix)
                
                log_clustering(dataset['name'], eval_metrics, "with global codebook of size "+str(codebook_size))
                
                dataset['sound_ids'] = sound_ids      # order of the sounds
                dataset['labels'] = true_labels       # idx
                dataset['label_ids'] = label_ids      # audioset id
                dataset['label_names'] = label_names  # name
                dataset['X_{}'.format(embedding_name)] = features
                dataset['codebook_{}'.format(embedding_name)] = codebook_global
                dataset['encoded_{}'.format(embedding_name)] = encoded_audios
                dataset['histograms_{}'.format(embedding_name)] = histograms
                dataset['labels_{}'.format(embedding_name)] = cluster_labels
                dataset['evaluation_metrics_{}'.format(embedding_name)] = eval_metrics

                web_visu_dataset_name = '-'.join([dataset['name'], 'global_codebook', str(codebook_size)])
                dataset['web_visu_dataset_name'] = web_visu_dataset_name
                save_graph_json_for_web_visu(graph, sound_ids, web_visu_dataset_name, embedding_name)
            
                #save_graph_json_for_web_visu(graph, sound_ids, dataset['name'], embedding_name)

        filename_to_save = "files/"+str(codebook_size)+"_global_codebook_clusters.file"
        save_as_binary(datasets, filename_to_save)
        add_clustering_info_for_web_visu(datasets)



def local_codebook_encoding_main(codebook_sizes, datasets, ontology_by_id):
    for codebook_size in codebook_sizes:       
        for dataset in datasets:
            sound_ids, true_labels, label_ids, label_names = create_label_vector(dataset['dataset'], ontology_by_id) 
            for embedding_name, embedding_folder in EMBEDDING_FOLDERS.items():
                features = load_features(sound_ids, embedding_folder)

                if(not have_enough_data_to_cluster(features, codebook_size)):
                    print("\nNot enough data to process codebook for ", codebook_size, " words, in dataset: ", dataset['name'], "\n\n")
                    dataset = None
                else:
                    codebook = generate_codebooks(features, codebook_size)
                    encoded_audios = encode_frames(features, codebook)
                    histograms = create_histograms(codebook, encoded_audios) # maybe not really needed, only for plotting
                    
                    similarity_matrix = compute_similarity_matrix_from_tfidf(encoded_audios)
                    cluster_labels, eval_metrics, graph = cluster_dataset(true_labels, similarity_matrix)
                    
                    log_clustering(dataset['name'], eval_metrics, "with local codebook of size "+str(codebook_size))
                    
                    dataset['sound_ids'] = sound_ids      # order of the sounds
                    dataset['labels'] = true_labels       # idx
                    dataset['label_ids'] = label_ids      # audioset id
                    dataset['label_names'] = label_names  # name
                    dataset['X_{}'.format(embedding_name)] = features
                    dataset['codebook_{}'.format(embedding_name)] = codebook
                    dataset['encoded_{}'.format(embedding_name)] = encoded_audios
                    dataset['histograms_{}'.format(embedding_name)] = histograms
                    dataset['labels_{}'.format(embedding_name)] = cluster_labels
                    dataset['evaluation_metrics_{}'.format(embedding_name)] = eval_metrics

                    web_visu_dataset_name = '-'.join([dataset['name'], 'local_codebook', str(codebook_size)])
                    dataset['web_visu_dataset_name'] = web_visu_dataset_name
                    save_graph_json_for_web_visu(graph, sound_ids, web_visu_dataset_name, embedding_name)
            
                    #save_graph_json_for_web_visu(graph, sound_ids, dataset['name'], embedding_name)

        filename_to_save = "files/"+str(codebook_size)+"_codebook_clusters.file"
        save_as_binary(datasets, filename_to_save)
        add_clustering_info_for_web_visu(datasets)



def statistical_agg_main(datasets, ontology_by_id):
    #========= original pipelines ======================================= 
    for dataset in datasets:
        sound_ids, true_labels, label_ids, label_names = create_label_vector(dataset['dataset'], ontology_by_id)
        for embedding_name, embedding_folder in EMBEDDING_FOLDERS.items():
            features = load_features(sound_ids, embedding_folder)
            #dataset = create_merged_features(dataset) #==> needs openl3?
            features_mean = statistical_aggregation_features(features)
            similarity_matrix = compute_similarity_matrix(features_mean)
            cluster_labels, eval_metrics, graph = cluster_dataset(true_labels, similarity_matrix)

            log_clustering(dataset['name'], eval_metrics, "with original setup")
                    
            dataset['sound_ids'] = sound_ids      # order of the sounds
            dataset['labels'] = true_labels       # idx
            dataset['label_ids'] = label_ids      # audioset id
            dataset['label_names'] = label_names  # name
            
            dataset['X_mean_{}'.format(embedding_name)] = features_mean
            dataset['X_{}'.format(embedding_name)] = features
            dataset['labels_{}'.format(embedding_name)] = cluster_labels
            dataset['evaluation_metrics_{}'.format(embedding_name)] = eval_metrics
            
            #web-visu
            #web_visu_dataset_name = dataset['name'] + ' mean'
            #web_visu_dataset_name = 'test-mfcc'
            #web_visu_dataset_name = '{}-{}'.format(dataset['name'], embedding_name)

            web_visu_dataset_name = '-'.join([dataset['name'], 'mean'])
            dataset['web_visu_dataset_name'] = web_visu_dataset_name
            save_graph_json_for_web_visu(graph, sound_ids, web_visu_dataset_name, embedding_name)
        
    filename_to_save = "files/mean_computed_datasets_clusters.file"
    save_as_binary(datasets, filename_to_save)
    add_clustering_info_for_web_visu(datasets)



if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        #json.dump({}, open('web-visu/clustering_info.json', 'w')) #reset web-visu
     
        if(len(sys.argv)<=1): #if any argument, don't compute and only load previous json:
            # load ontology
            ontology = json.load(open('json/ontology.json', 'rb'))
            ontology_by_id = {obj['id']: obj for obj in ontology}

            # load datasets
            dataset_files = os.listdir('datasets')
            datasets = [json.load(open('datasets/'+f, 'rb')) for f in dataset_files]

            # cleaning
            #remove_failed_embeddings(datasets) # NOTE: when using mffc we don't need this step anymore

            codebook_sizes = [64, 128, 256, 512, 1024, 2048]
            #codebook_sizes = [1024, 2048]
            #codebook_sizes = [2] # develop audioset
            #codebook_sizes = [1024, 2048]
            #codebook_sizes = [64] #test
            
            
            print("\n CAREFUL CODEBOOK SIZES")
            # 3 different pipelines at the moment:
            #global_codebook_encoding_main(codebook_sizes, datasets, ontology_by_id)
            local_codebook_encoding_main(codebook_sizes, datasets, ontology_by_id)
            statistical_agg_main(datasets, ontology_by_id)

        #TODO: adapt to many different clusterings now
        #TODO: Adapt web-visu to many codebooks
        #add_clustering_info_for_web_visu(datasets)