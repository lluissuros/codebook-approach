# Clustering of multiple-event online sound collections with the codebook approach
This is the accompanying codebase to my Master Thesis in SMC

## Custom dataset: acoustic scenes
https://drive.google.com/open?id=1LzMi7FHU5lUZxLiCl1jefBxIBVZG8xGv6mx4czmhm6Y


## freesound_processing_pipeline folder

Given a dataset of feature vectors series, the codebook will be generated. This codebook will be used to encode the feature vectors, so each frame will be encoded as its nearest codeword in the feature space. This encoded vector can be treated analogous to a text, where each codeword is analog to a natural language word. In our particular processing pipeline, TF-IDF is performed in order to reward less common codewords. Since the output of TF-IDF is a fixed-sized vector, a similarity matrix can be computed. This similarity matrix is used by the clustering algorithm to finally obtain the clusters. 

The presented code is modular enough to allow the use of different NLP algorithms in order to produce alternative similarity matrices. For example, histogram intersection was used in a previous iteration instead of TF-IDF. Different clustering techniques are also easy to plug-in (as an example, k-means was previously used in an earlier iteration before switching to k-nn graph-based clustering which proved to be much more faster and easier to render in the web browser).

In this graph, each node corresponds to a Freesound clip.

Clusters in the graph are identified using a Louvain community detection algorithm [implementation](https://github.com/taynaud/python-louvain/tree/networkx2) with the [NetworkX](https://networkx.github.io/) Python package.

Setup
-------------------
- Install dependencies in a virtual environment:
  ```
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

- Copy the `settings.example.py` file to `settings.py`, open it, and follow the instructions. Here we basically configure were the audio features of the sound files have to be located in the computer.
  ```
  cp settings.example.py settings.py
  ```

- As you might now understand, you need to have a folder containing the pre-calculated audio features for the sounds in the different datasets. There are in total around 30k sounds (with freesound IDs in `all_sound_ids.json`) and 45 datasets (JSON files in `datasets/`).

- You can start the clusterings by typing:
  ```
  python clustering.py
  ```
  This will output some results in the console. (TODO: add stats e.g. num clusters, num sounds, ...).
  It will also save the clustered graph so that we can visualise them in a 2D representation.

- You can start the visualisation web server by typing:
  ```
  python web-visu/start_server.py
  ```
  Then you can access the web app from your browser at: `http://localhost:8100/web-visu/`.


