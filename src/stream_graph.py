# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import time
import random
from collections import defaultdict
from scipy.sparse import coo_matrix

import dgl
from dgl.data.utils import save_graphs
import torch

from data_loader import DataLoader
from utils import *
from temporal_sampler import temporal_sampler


class StreamGraph():
    ''' A stream graph is a dynamic graph that can be defined by :math: `G = (V, E, T)`, where :math:`T` provides information about
        the time at which an edge occured between two nodes. Stream graphs are built upon `dgl` objects, where time is stored on each
        edges within a `timestamp` field. 
        
        Parameters
        -----------
            data: DataLoader
                DataLoader object containing a `DataFrame` object with a list triplets :math:`(u, v, t)` for each edge
                between nodes :math:`u` and :math:`v` at time :math:`t`. '''

    def __init__(self, data: DataLoader):
        self.name = data.name
        self.IN_PATH = data.OUT_PATH
        self.data_df = data.data_df
        self.add_self_edges = False
        self.add_temporal_edges = False
        self.neg_sampling = False
        self.is_splitted = False
        self.batches = False
        
        #data_df = self.__load_preprocessed_data(self.name)
        print('Creating stream graph ...')
        self.g = dgl.graph((self.data_df['src'], self.data_df['dest']))
        self.g.edata['timestamp'] = torch.from_numpy(self.data_df['t'].to_numpy())


    def __load_preprocessed_data(self, dataset: str):
        return pd.read_pickle(f'{self.IN_PATH}/{dataset}.pkl')

    
    def directed2undirected(self, copy_ndata: bool = True, copy_edata: bool = True):
        ''' DGLGraph is directed by default. In order to perform message passing in both ways between two nodes, it is
            necessary to convert directed graph to undirected, by adding reverse edges. '''

        self.g = dgl.add_reverse_edges(self.g, copy_ndata=copy_ndata, copy_edata=copy_edata)
        if self.is_splitted:
            self.train_g = dgl.add_reverse_edges(self.train_g, copy_ndata=copy_ndata, copy_edata=copy_edata)
            self.train_pos_g = dgl.add_reverse_edges(self.train_pos_g, copy_ndata=copy_ndata, copy_edata=copy_edata)
            self.val_pos_g = dgl.add_reverse_edges(self.val_pos_g, copy_ndata=copy_ndata, copy_edata=copy_edata)
            self.test_pos_g = dgl.add_reverse_edges(self.test_pos_g, copy_ndata=copy_ndata, copy_edata=copy_edata)
            self.test_pos_seen_g = dgl.add_reverse_edges(self.test_pos_seen_g, copy_ndata=copy_ndata, copy_edata=copy_edata)
        if self.neg_sampling:
            self.train_neg_g = dgl.add_reverse_edges(self.train_neg_g, copy_ndata=copy_ndata, copy_edata=copy_edata)
            self.val_neg_g = dgl.add_reverse_edges(self.val_neg_g, copy_ndata=copy_ndata, copy_edata=copy_edata)
            self.test_neg_g = dgl.add_reverse_edges(self.test_neg_g, copy_ndata=copy_ndata, copy_edata=copy_edata)
            self.test_neg_seen_g = dgl.add_reverse_edges(self.test_neg_seen_g, copy_ndata=copy_ndata, copy_edata=copy_edata)


    def _reindex_nodes_time(self):
        ''' Creates a dictionary with reindexed nodes from 0 to :math:`|V|-1`, using the orginial DataFrame. 
            Output
            -------
                res : dict '''

        res = {}
        val = 0
        for i, j, t in zip(self.data_df['src'], self.data_df['dest'], self.data_df['t']):
            if f'{i}_{t}' not in res:
                res[f'{i}_{t}'] = val
                val += 1
            if f'{j}_{t}' not in res:
                res[f'{j}_{t}'] = val
                val += 1
        return res


    def __add_temporal_edges_graph(self, g, timerange):
        ''' Given a `dgl` graph, builds a postive and negative `PDAG` by adding temporal edges. Temporal edge :math:`E(u_t, u_{t+\delta})` is
            a directed link between node :math:`u` at time :math:`t`, and node :math:`u` at time :math:`t+\delta`, with :math:`\delta` the number
            of minimal timestep interval between two consecutives events in the dataset.

            Parameters
            -----------
                g : `dgl` graph
                timerange : np.ndarray
                    Array of all possible timesteps covered by graph g.

            Output
            -------
                g_pdag, g_pdag_neg : `dgl` graphs
                    Graphs with respectively positive and negative edges, as well as temporal edges, for the whole timerange period. '''

        # Dictionary with |V| keys, and last k time indexes at which node was seen.
        # This dictionary is useful to gather embeddings of all nodes at last k timesteps.
        self.last_k_emb_idx = defaultdict(list)

        # Reindex all nodes according to time
        time_idx_nodes = self._reindex_nodes_time()

        # Build PDAG
        list_timerange = list(timerange)
        min_t = int(g.edata['timestamp'].min())
        max_t = int(g.edata['timestamp'].max())
        df = self.data_df[(self.data_df['t']<max_t) & (self.data_df['t']>=min_t)]

        unique_t = df['t'].unique()
        rows = np.array([], dtype=np.int16)
        cols = np.array([], dtype=np.int16)
        rows_neg = np.array([], dtype=np.int16)
        cols_neg = np.array([], dtype=np.int16)

        prev_nodes = set()
        prev_t = 0

        for idx, t in enumerate(unique_t):

            df_t = df[df['t']==t].copy()

            # -------- Positive edges ---------
            src_t = np.array(df_t['src'])
            dest_t = np.array(df_t['dest'])

            # Directed to undirected
            src_t_temp = src_t
            dest_t_temp = dest_t
            src_t = np.append(src_t, dest_t_temp)
            dest_t = np.append(dest_t, src_t_temp)

            curr_nodes = set(src_t).union(set(dest_t))
            # Save time indexes for seen nodes
            for node in curr_nodes:
                self.last_k_emb_idx[node].append(time_idx_nodes[f'{node}_{t}']) 
            
            temporal_nodes = curr_nodes.intersection(prev_nodes)

            # Fill COO adjacency matrix with time-indexed nodes
            rows = np.append(rows, [time_idx_nodes[f'{src}_{t}'] for src in src_t])
            cols = np.append(cols, [time_idx_nodes[f'{dest}_{t}'] for dest in dest_t])

            # Fill COO adjacency matrix with temporal nodes (directed from past to current time)
            if len(temporal_nodes) > 0:
                rows = np.append(rows, [time_idx_nodes[f'{node}_{prev_t}'] for node in temporal_nodes])
                cols = np.append(cols, [time_idx_nodes[f'{node}_{t}'] for node in temporal_nodes])


            # -------- Negative edges ---------
            # If only 2 nodes in graph at time t, we cannot create negative edges
            pos_edge_list = [(u, v) for u, v in zip(src_t, dest_t)]

            if len(curr_nodes) > 2:
                len_src_t = len(src_t)

                for src_node in src_t[:int(len_src_t/2)]:
                    curr_nodes_list = list(curr_nodes)
                    curr_nodes_list.remove(src_node) # Self edge is not allowed
                    
                    for i in range(3):
                        neg_edge = (src_node, random.choice(curr_nodes_list)) 
                        if neg_edge not in pos_edge_list:
                            rows_neg = np.append(rows_neg, [time_idx_nodes[f'{src}_{t}'] for src in neg_edge])
                            cols_neg = np.append(cols_neg, [time_idx_nodes[f'{dest}_{t}'] for dest in neg_edge[::-1]])
                            break

            # Fill COO adjacency matrix with temporal nodes (directed from past to current time)
            if len(temporal_nodes) > 0:
                rows_neg = np.append(rows_neg, [time_idx_nodes[f'{node}_{prev_t}'] for node in temporal_nodes])
                cols_neg = np.append(cols_neg, [time_idx_nodes[f'{node}_{t}'] for node in temporal_nodes])

            # Update previous time variables with current values
            prev_nodes = curr_nodes
            prev_t = t
        
        # Create data arrays (no weights)
        data = np.ones(len(rows), dtype=np.int16)
        data_neg = np.ones(len(rows_neg), dtype=np.int16)

        # Build graphs from COO adjacency matrix
        coo_m = coo_matrix((data, (rows, cols)))

        # Ensure that rows and cols have the same max
        max_rows_neg = max(rows_neg)
        max_cols_neg = max(cols_neg)
        if max_rows_neg > max_cols_neg:
            rows_neg = np.append(rows_neg, max_cols_neg)
            cols_neg = np.append(cols_neg, max_cols_neg)
            data_neg = np.append(data_neg, 1)
        else:
            cols_neg = np.append(cols_neg, max_rows_neg)
            rows_neg = np.append(rows_neg, max_cols_neg)
            data_neg = np.append(data_neg, 1)
        coo_m_neg = coo_matrix((data_neg, (rows_neg, cols_neg)))

        # Ensure that pos and neg have the same max
        max_pos = max(max(rows), max(cols))
        max_neg = max(max(rows_neg), max(cols_neg))
        if max_pos > max_neg:
            rows_neg = np.append(rows_neg, max_pos)
            cols_neg = np.append(cols_neg, max_pos)
            data_neg = np.append(data_neg, 1)
            coo_m_neg = coo_matrix((data_neg, (rows_neg, cols_neg)))
        else:
            rows = np.append(rows, max_neg)
            cols = np.append(cols, max_neg)
            data = np.append(data, 1)
            coo_m = coo_matrix((data, (rows, cols)))

        print('coo pos shape : ', coo_m.shape)
        print('coo neg shape : ', coo_m_neg.shape)
        
        g_pdag = dgl.from_scipy(coo_m)
        g_pdag_neg = dgl.from_scipy(coo_m_neg)

        print(f'    Ratio #negative edges / #positive edges : {len(rows_neg)/len(rows):.4f}')

        return g_pdag, g_pdag_neg


    def __add_temporal_edges(self):
        ''' If :math:`u(t)` and :math:`u(t+\delta)` both exist, adds a directed link between :math:`u(t)` and :math:`u(t+\delta)`. ''' 

        self.add_temporal_edges = True

        #self.g = self.__add_temporal_edges_graph2(self.g, self.trange)
        if self.is_splitted:
            #self.train_g, _ = self.__add_temporal_edges_graph(self.train_g, self.trange_train)
            self.train_pos_g, self.train_neg_g = self.__add_temporal_edges_graph(self.train_pos_g, self.trange_train)
            self.train_g = self.train_pos_g
        #    self.val_pos_g = self.__add_temporal_edges_graph(self.val_pos_g, self.trange_val)
        #    self.test_pos_g, self.test_neg_g = self.__add_temporal_edges_graph(self.test_pos_g, self.trange_test)
        #    self.test_pos_seen_g = self.__add_temporal_edges_graph(self.test_pos_seen_g, self.trange_test)


    def compute_features(self, feat_struct: str, add_self_edges=True, normalized=True, timestep=20):
        ''' Attach features to existing graph(s). Features are computed from adjacency matrix of the graph.
        
            Parameters
            -----------
                feat_struct: str
                    Available data structures are :
                    * `agg` : Average adjacency matrix over time. The size of the feature matrix is :math:`(|V|, |V|)`. 
                            Each entry in the matrix is computed as :math:`f_{i, j}=\dfrac{1}{|T|}\sum_{t \in T}a_{i, j}(t)`. 
                            Note that with this structure, a lot of temporal information is lost.
                    * `time_tensor` : The adjacency matrix is considered using a **3D-tensor** with size :math:`(|V|, |V|, T)`, with
                            :math:`T` the lenght of time sequence. Adjacency matrix is not normalized. 
                    * `temporal_edges` : Adjacency matrix considering each node as a tuple :math:`(u, t)` where :math:`t` is the time
                            at which node :math:`u` is present. 
                add_self_edges: bool (default=True) 
                    If True, add self-edges for each node in graph. 
                normalized : bool (default=True)
                    If True, normalize adjacency matrix such as :math:`\tild{A}=D^{-1}AD^{-1}` with :math:`D` the matrix of degrees. '''
                            
        print('\nCompute features ...')
        print(f'    Data structure : {feat_struct}')
        self.add_self_edges = add_self_edges

        if feat_struct=='temporal_edges':

            # Create graphs with temporal edges
            self.__add_temporal_edges()

            # Compute features as coo adjacency matrix 
            adj = self.train_pos_g.adj()

        elif feat_struct=='time_tensor':

            # Build training adjacency 3d-tensor
            max_train_t = int(self.train_pos_g.edata['timestamp'].max())
            df_train = self.data_df[self.data_df['t']<=max_train_t]
            train_trange = list(np.arange(int(self.train_pos_g.edata['timestamp'].min()), int(self.train_pos_g.edata['timestamp'].max()) + timestep, timestep))
            train_trange_idx = [i for i in range(len(train_trange))]
            adj = torch.zeros(self.train_pos_g.number_of_nodes(), self.train_pos_g.number_of_nodes(), len(train_trange_idx))

            # Fill 3d-tensor with 1 if edge between u and v at time t exists
            for node_src, node_dest, t in zip(df_train['src'], df_train['dest'], df_train['t']):
                t_index = train_trange.index(t)
                adj[node_src, node_dest, t_index] = 1

            # Add self-edges over time
            if add_self_edges:
                for node in self.train_pos_g.nodes():
                    adj[node, node, :] = torch.ones(len(train_trange_idx))

        elif feat_struct=='DTFT':

            # Build training adjacency 3d-tensor
            max_train_t = int(self.train_pos_g.edata['timestamp'].max())
            df_train = self.data_df[self.data_df['t']<=max_train_t]
            train_trange = list(np.arange(int(self.train_pos_g.edata['timestamp'].min()), int(self.train_pos_g.edata['timestamp'].max()) + timestep, timestep))
            train_trange_idx = [i for i in range(len(train_trange))]
            adj = torch.zeros(self.train_pos_g.number_of_nodes(), self.train_pos_g.number_of_nodes(), len(train_trange_idx))

            # Fill 3d-tensor with 1 if edge between u and v at time t exists
            for node_src, node_dest, t in zip(df_train['src'], df_train['dest'], df_train['t']):
                t_index = train_trange.index(t)
                adj[node_src, node_dest, t_index] = 1

            N = len(train_trange_idx)
            fourier_feat = torch.zeros(self.train_pos_g.number_of_nodes(), self.train_pos_g.number_of_nodes(), N)
            for node_src in range(adj.shape[0]):
                for node_dest in range(adj.shape[1]):            
                    X_temp = np.fft.fft(adj[node_src, node_dest, :], N)
                    fourier_feat[node_src, node_dest, :] = torch.from_numpy(np.abs(X_temp))
            
            adj = fourier_feat.clone()

        elif feat_struct=='agg':

            if self.batches:

                # Positive batches
                for train_batch_g in self.train_pos_batches:
                    train_batch_timerange = np.arange(int(train_batch_g.edata['timestamp'].min()), int(train_batch_g.edata['timestamp'].max()) + timestep, timestep)
                    # Compute features 
                    if train_batch_timerange[0] != 0:
                        train_batch_feat = compute_agg_features(train_batch_g, train_batch_timerange, add_self_edges=add_self_edges) 
                        if normalized:
                            train_batch_g.ndata['feat'] = normalize_adj(train_batch_feat)
                        else:
                            train_batch_g.ndata['feat'] = train_batch_feat
                # Negative batches
                for train_neg_batch_g in self.train_neg_batches:
                    train_batch_timerange = np.arange(int(train_neg_batch_g.edata['timestamp'].min()), int(train_neg_batch_g.edata['timestamp'].max()) + timestep, timestep)
                    # Compute features 
                    if train_batch_timerange[0] != 0:
                        train_neg_batch_feat = compute_agg_features(train_neg_batch_g, train_batch_timerange, add_self_edges=add_self_edges) 
                        if normalized:
                            train_neg_batch_g.ndata['feat'] = normalize_adj(train_neg_batch_feat)
                        else:
                            train_neg_batch_g.ndata['feat'] = train_neg_batch_feat
            
            else:
                adj = compute_agg_features(self.train_pos_g, self.trange_train, add_self_edges=add_self_edges)
            
                # Normalize adjacency
                if normalized:
                    adj = normalize_adj(adj)

        elif feat_struct=='agg_simp':

            if self.batches:

                # Positive batches
                for train_batch_g in self.train_pos_batches:
                    train_batch_timerange = np.arange(int(train_batch_g.edata['timestamp'].min()), int(train_batch_g.edata['timestamp'].max()) + timestep, timestep)
                    # Compute features 
                    if train_batch_timerange[0] != 0:
                        train_batch_feat = compute_agg_features_simplified(train_batch_g, train_batch_timerange, add_self_edges=add_self_edges) 
                        if normalized:
                            train_batch_g.ndata['feat'] = normalize_adj(train_batch_feat)
                        else:
                            train_batch_g.ndata['feat'] = train_batch_feat
                # Negative batches
                for train_neg_batch_g in self.train_neg_batches:
                    train_batch_timerange = np.arange(int(train_neg_batch_g.edata['timestamp'].min()), int(train_neg_batch_g.edata['timestamp'].max()) + timestep, timestep)
                    # Compute features 
                    if train_batch_timerange[0] != 0:
                        train_neg_batch_feat = compute_agg_features_simplified(train_neg_batch_g, train_batch_timerange, add_self_edges=add_self_edges) 
                        if normalized:
                            train_neg_batch_g.ndata['feat'] = normalize_adj(train_neg_batch_feat)
                        else:
                            train_neg_batch_g.ndata['feat'] = train_neg_batch_feat
            
            else:
                adj = compute_agg_features(self.train_pos_g, self.trange_train, add_self_edges=add_self_edges)
            
                # Normalize adjacency
                if normalized:
                    adj = normalize_adj(adj)

        # Attach features to graph nodes
        if not self.batches:
            self.train_g.ndata['feat'] = adj
            self.train_pos_g.ndata['feat'] = adj

        print('Done !')


    def create_batches(self, batch_size, timestep=20):
        ''' Creates lists of batches containing positive and negative edges for training graph. Each batch contains `batch_size` number
            of timesteps. 
            
            Parameters
            -----------
                batch_size : int
                    Number of timesteps in each batch. '''

        print('\nCreating batches ...')
        self.batches = True

        # Build graph batches
        train_pos_batches, indexes_pos = temporal_sampler(self.train_pos_g, batch_size, timestep)
        train_neg_batches, _ = temporal_sampler(self.train_neg_g, batch_size, timestep)

        # Filter graphs only for the period where a positive graph is computed
        # Note : it is not possible to create Numpy arrays of DGL Graphs
        self.train_pos_batches = []
        self.train_neg_batches = []
        for g_pos, g_neg, mask in zip(train_pos_batches, train_neg_batches, indexes_pos):
            if mask:
                self.train_pos_batches.append(g_pos)
                self.train_neg_batches.append(g_neg)

        print('Done !')


    def train_test_split(self, val_size: float, test_size: float, timestep: int, neg_sampling: bool = True, metric=None):

        print('\nSplitting graph ...')

        self.is_splitted = True
        self.neg_sampling = neg_sampling

        # Compute validation time and test time
        val_cut = 1 - val_size - test_size
        test_cut = 1 - test_size
        val_time, test_time = np.quantile(self.g.edata['timestamp'], [val_cut, test_cut])

        # If dataset is 'HighSchool', train test split is fixed automatically to fit days in data
        if self.name == 'HighSchool':
            val_time = 1386226820
            test_time = 1386259200
            print(f'For this dataset, val and test lengths have been forced to fixed sizes.')

        # Split edge set for training, validation and test
        # Edges are divided into 2 groups : positive (link in graph) and negative (no link in graph)
        source, dest = self.g.edges()
        timestamp = self.g.edata['timestamp']

        # Masks for datasets filtering
        train_mask = self.g.edata['timestamp'] <= val_time
        val_mask = torch.logical_and(self.g.edata['timestamp'] > val_time, self.g.edata['timestamp'] <= test_time)
        test_mask = self.g.edata['timestamp'] > test_time

        eids = np.arange(self.g.number_of_edges())
        val_nb_edge = len(source[val_mask])
        test_nb_edge = len(source[test_mask])
        train_nb_edge = len(eids) - test_nb_edge - val_nb_edge

        # -- Positive edges --
        # Postiive edges are used to create positive graphs according to splits

        if metric is not None:
            split = metric.split('@')
            if len(split) == 2:
                nb_timesteps = int(split[1])
                nb_edges = nb_edges_at_ts(self.data_df, val_time, nb_timesteps)
            else:
                nb_edges = len(source)
        else:
            nb_edges = len(source)

        train_pos_g = dgl.graph((source[train_mask], dest[train_mask]), num_nodes=self.g.number_of_nodes())
        val_pos_g = dgl.graph((source[val_mask][:nb_edges], dest[val_mask][:nb_edges]), num_nodes=self.g.number_of_nodes())
        test_pos_g = dgl.graph((source[test_mask][:nb_edges], dest[test_mask][:nb_edges]), num_nodes=self.g.number_of_nodes())
        train_pos_g.edata['timestamp'] = timestamp[train_mask]
        val_pos_g.edata['timestamp'] = timestamp[val_mask][:nb_edges]
        test_pos_g.edata['timestamp'] = timestamp[test_mask][:nb_edges]

        # In addition to previous test set, we create a another test set which contains only edges
        # that appeared before in time (i.e in training set)
        seen_edges_tuple = [(int(src), int(dst)) for src, dst in zip(train_pos_g.edges()[0], train_pos_g.edges()[1])]
        test_edges_tuple = [(int(src), int(dst)) for src, dst in zip(test_pos_g.edges()[0], test_pos_g.edges()[1])]
        inter = list(set(test_edges_tuple).intersection(set(seen_edges_tuple)))
        test_eids = []
        for idx, tup in enumerate(test_edges_tuple):
            if tup in inter:
                test_eids.append(idx)
        test_pos_seen_g = dgl.edge_subgraph(test_pos_g, test_eids, preserve_nodes=True)
        test_seen_nb_edge = len(test_pos_seen_g.edges()[0])

        if neg_sampling:
            # -- Negative edges --
            # Negative edges are sampled randomly and used to create negative graphs according to splits
            edge_list_train = make_edge_list(source, dest, timestamp, train_mask)
            edge_list_val = make_edge_list(source, dest, timestamp, val_mask)
            edge_list_test = make_edge_list(source, dest, timestamp, test_mask)
            edge_list_seen_test = make_edge_list(test_pos_seen_g.edges()[0], test_pos_seen_g.edges()[1], test_pos_seen_g.edata['timestamp'], [True]*len(test_pos_seen_g.edata['timestamp']))
            timerange = np.arange(int(self.g.edata['timestamp'].min()), int(self.g.edata['timestamp'].max()), timestep)

            # Masks for negative edges according to splits
            train_mask_neg = timerange <= val_time
            val_mask_neg = np.logical_and(timerange > val_time, timerange <= test_time)
            test_mask_neg = timerange > test_time

            # Negative edges - Training set
            train_neg_edge = negative_sampling(self.g, timerange[train_mask_neg], edge_list_train)

            train_src_id, train_dest_id, train_t = zip(*train_neg_edge)
            train_neg_g = dgl.graph((train_src_id, train_dest_id), num_nodes=self.g.number_of_nodes())
            train_neg_g.edata['timestamp'] = torch.tensor(train_t)

            # Negative edges - Validation set
            val_neg_edge = negative_sampling(self.g, timerange[val_mask_neg], edge_list_val)

            val_src_id, val_dest_id, val_t = zip(*val_neg_edge)
            val_neg_g = dgl.graph((val_src_id, val_dest_id), num_nodes=self.g.number_of_nodes())
            val_neg_g.edata['timestamp'] = torch.tensor(val_t)

            # Negative edges - Test set
            test_neg_edge = negative_sampling(self.g, timerange[test_mask_neg], edge_list_test)

            test_src_id, test_dest_id, test_t = zip(*test_neg_edge)
            test_neg_g = dgl.graph((test_src_id, test_dest_id), num_nodes=self.g.number_of_nodes())
            test_neg_g.edata['timestamp'] = torch.tensor(test_t)

            # Negative edges - Test seen set
            test_neg_seen_edge = negative_sampling(self.g, timerange[test_mask_neg], edge_list_seen_test)

            test_src_id, test_dest_id, test_t = zip(*test_neg_seen_edge)
            test_neg_seen_g = dgl.graph((test_src_id, test_dest_id), num_nodes=self.g.number_of_nodes())
            test_neg_seen_g.edata['timestamp'] = test_pos_seen_g.edata['timestamp']

        # Build training graph (ie. graph without test and valid edges)
        train_g = dgl.remove_edges(self.g, eids[train_nb_edge:])

        # Store timeranges
        self.trange = np.arange(int(self.g.edata['timestamp'].min()), int(self.g.edata['timestamp'].max()) + timestep, timestep)
        self.trange_train = self.trange[self.trange <= val_time]
        self.trange_val = self.trange[(self.trange > val_time) & (self.trange <= test_time)]
        self.trange_test = self.trange[self.trange > test_time]
        self.edata_train = train_pos_g.edata['timestamp']
        self.edata_val = val_pos_g.edata['timestamp']
        self.edata_test = test_pos_g.edata['timestamp']

        # Save graphs
        if neg_sampling:
            graphs_to_save = [train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g, test_pos_seen_g, test_neg_seen_g]
            save_graphs(f'../preprocessed_data/data.bin', graphs_to_save)
            self.train_neg_g = train_neg_g 
            self.val_neg_g = val_neg_g
            self.test_neg_g = test_neg_g
            self.test_neg_seen_g = test_neg_seen_g
        else:
            graphs_to_save = [train_g, train_pos_g, val_pos_g, test_pos_g, test_pos_seen_g]
            save_graphs(f'../preprocessed_data/data.bin', graphs_to_save)

        print('Done !')
        self.train_g = train_g
        self.train_pos_g = train_pos_g
        self.val_pos_g = val_pos_g
        self.test_pos_g = test_pos_g
        self.test_pos_seen_g = test_pos_seen_g

    def rank_edges(self, df, timerange, metric=None, timestep=20):
        ''' Given a link stream dataframe and a timerange, ranks all edges appearing within the timerange
            using their order of appearance. Output links and their corresponding ranks as a triplet of
            arrays. Note that duplicated edges are not removed at this step. '''

        min_t = np.min(timerange) - timestep

        if metric is not None and '@' in metric:
            nb_edges = nb_edges_at_ts(df, min_t, int(metric.split('@')[1]))
            df_tmp = df[(df['t'] >= min_t)][['src', 'dest']]
            df_filtered = df_tmp.iloc[:nb_edges, :]
        else:
            max_t = np.max(timerange)
            df_filtered = df[(df['t'] >= min_t) & (df['t'] <= max_t)][['src', 'dest']]

        # Save df train
        max_train_t = self.trange_train.max()
        df_train = self.data_df[self.data_df['t']<=max_train_t]
        df_train.to_pickle('logs/SF2H/SF2H_train.pkl', protocol=3)
        # --------

        e_src = df_filtered['src'].values
        e_dest = df_filtered['dest'].values
        e_dup = np.array(df_filtered.duplicated(keep='first'))
        e_ranks = df_filtered.reset_index().index.values

        return e_src, e_dest, e_ranks, e_dup
        

