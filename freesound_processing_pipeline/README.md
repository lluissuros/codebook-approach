# Freesound Graph Clustering

Freesound Graph Clustering is a repository hosting some code for evaluating different features for the unsupervised classification (clustering) of audio content from [Freesound](https://freesound.org/).
The clustering performance is evaluated on different datasets using an external cluster validation, which consists in comparing the results of a clustering to an externally known result (provided ground truth labels). The data has been collected within the [Freesound Annotator](https://annotator.freesound.org/) platform, which hosts the [FSD](https://annotator.freesound.org/fsd/explore/) dataset containing sounds organised following the [AudioSet Ontology](https://research.google.com/audioset/ontology/index.html).

The clustering method consists on partitioning a k-nearest neighbors graph.
In this graph, each node corresponds to a Freesound clip.
Edges between nodes reflect a notion of content-based similarity calculated on a feature space, for instance, using the [AudioSet](https://research.google.com/audioset/) embeddings.
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


