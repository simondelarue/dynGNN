# Learning on dynamic graphs using Graph Neural Networks

In this repository, we explore several approaches to (i) represent dynamic graphs and (ii) learn on these representations through embedding methods, using deep-learning based approaches such as graph neural networks (GNNs) among others.
We evaluate performance of the different combinations of data-structures and models through the analysis of underlying machine-learning tasks on real-world datasets.

This repository provides a framework for:
* loading data
* representing dynamic graph
* training models on these representations
* evaluating performance on machine learning tasks

**Representations**  
We explore different methods to encode dynamic graphs. We divide these methods into two categories:
- static representations, where temporal information is aggregated. The dynamic problem is thus transformed into a static one
- dynamic representations, where the goal is to encode as much information as possible into one single structure

**Models**  
In order to learn from and on temporal data, we use state-of-the-art GNN models, as well as custom models specifically designed to fit complex dynamic graph representations.

## Table of contents

1. [Setup](#Setup)  
2. [Usage](#Usage)  
3. [Datasets](#Datasets)  

## 1. Setup <a class="anchor" id="Setup"></a>

### Install dependencies
```bash
pip install requirements.txt
```

## 2. Usage <a class="anchor" id="Usage"></a>

The command ```python src/main.py``` is the entry-point for running the whole process. Different arguments are available to specify dataset, representation, model, machine learning task or evaluation metric.

**Arguments**
``` system
--data              Data source {SF2H, HighSchool, ia-contact, ia-contacts_hypertext2009, fb-forum, ia-enron-employees}
--cache             Path for split graphs if already cached
--feat_struct       Data structure {agg, agg_simp, time_tensor, temporal_edges, DTFT, baseline}
--step_prediction   Only available for 'temporal_edges' feature structure {single, multi}
--normalized        Consider both-sides normalized adjacency matrix {True, False}
--model             Graph Neural Network model {GraphConv, GraphSage, GCNTime, baseline_avg}
--batch_size        If batch size greater than 0, dynamic graph is split into batches
--emb_size          Node embedding size
--epochs            Number of epochs for training
--lr                Learning rate during training
--metric            Evaluation metric {auc, kendall, wkendall, spearmanr, {kendall, wkendall, spearmanr}@{5, 10, 25, 50, 100}}
--duplicate_edges   If True, allows duplicate edges in training graph {True, False}
--test_agg          If true, predictions are performed on a static graph test {True, False}
--predictor         Similarity function {dotProduct, cosine}
--loss_func         Loss function {BCEWithLogits, graphSage, marginRanking, torchMarginRanking, pairwise}
--shuffle_test      If True, shuffle test set links order {True, False}
```

**Example**

For example, the following command allows to:
- load ```SF2H``` dataset
- build a dynamic graph represented as a aggregated adjacency matrix
- learn node embeddings using *GraphSage* model (using *marginRanking* loss with *dot product* as similarity measure)
- evaluate performance on **learning to rank** task for 5 timesteps, using Kendall <img src="https://render.githubusercontent.com/render/math?math=\tau"> metric, on an aggregated test graph

```bash
python3 src/main.py --data SF2H --feat_struct agg_simp --model GraphSage --epoch 600 --lr 0.01 --metric kendall@5 --test_agg True --predictor dotProduct --loss_func marginRanking
```

For more examples, see ```run_results.sh```.

## 3. Datasets <a class="anchor" id="Datasets"></a>

In this work we use real-world datasets, where interactions between entities are encoded through *events* in the form of triplets <img src="https://render.githubusercontent.com/render/math?math=(t, uv)">, where <img src="https://render.githubusercontent.com/render/math?math=u"> and <img src="https://render.githubusercontent.com/render/math?math=v"> represent nodes in the graph, and <img src="https://render.githubusercontent.com/render/math?math=t"> is the time at which these nodes interacted with each other. We use the following datasets:

| Dataset      |
|--------------|
| [```SF2H```](http://www.sociopatterns.org/datasets/sfhh-conference-data-set/) |
| [```HighSchool```](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/) |  
| [```ia-contact```](https://networkrepository.com/ia-contact.php) | 
| [```ia-contacts-hypertext2009```](http://www.sociopatterns.org/datasets/hypertext-2009-dynamic-contact-network/) |
| [```ia-enron-employees```](https://networkrepository.com/ia_enron_employees.php) |