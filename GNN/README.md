# Dynamic graphs : node embedding using Graph Neural Networks

## Introduction

In this repository, we explore several approaches to model dynamic graphs and learn on them, using message passing schemes.
Performances of the different combinations of data-structures and models are compared through the analysis of underlying machine-learning tasks, such as link prediction or node classification.

## Usage

Below, an example of command to :
* load data (preprocess it if needed)
* define data structure
* compute features 
* train graph neural network model
* evaluate performance using specific metric 

``` bash
python3 src/main.py --data SF2H --feat_struct agg --normalized true --model GCN_lc --batch_size 30 --epoch 20 --lr 0.1 --metric auc
```

### Flags

``` system
--data              Data source {SF2H, HighSchool}
--cache             Path for splitted graphs if already cached
--feat_struct       Data structure {agg, time_tensor, temporal_edges}
--step_prediction   Only available for 'temporal_edges' feature structure {single, multi}
--normalized        Consider both-sides normalized adjacency matrix 
--model             Graph Neural Network model {GraphConv, GraphSage, GCNTime}
--batch_size        If batch size greater than 0, dynamic graph is splitted into batches
--emb_size          Node embedding size
--epochs            Number of epochs for training
--lr                Learning rate during training
--metric            Evaluation metric {auc}
```



