import pandas as pd
import numpy as np
import dgl
import torch
import os
from os.path import exists
from scipy.sparse import coo_matrix
from data_loader import DataLoader
from utils import *
from itertools import permutations
from dgl.data.utils import load_graphs, save_graphs

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

    def __add_self_edges(self):
        self.add_self_edges = True
        self.g.add_edges(self.g.nodes(), self.g.nodes())
        if self.is_splitted:
            self.train_g.add_edges(self.train_g.nodes(), self.train_g.nodes())
            self.train_pos_g.add_edges(self.train_pos_g.nodes(), self.train_pos_g.nodes())
            self.val_pos_g.add_edges(self.val_pos_g.nodes(), self.val_pos_g.nodes())
            self.test_pos_g.add_edges(self.test_pos_g.nodes(), self.test_pos_g.nodes())
            self.test_pos_seen_g.add_edges(self.test_pos_seen_g.nodes(), self.test_pos_seen_g.nodes())
        if self.neg_sampling:
            self.train_neg_g.add_edges(self.train_neg_g.nodes(), self.train_neg_g.nodes())
            self.val_neg_g.add_edges(self.val_neg_g.nodes(), self.val_neg_g.nodes())
            self.test_neg_g.add_edges(self.test_neg_g.nodes(), self.test_neg_g.nodes())
            self.test_neg_seen_g.add_edges(self.test_neg_seen_g.nodes(), self.test_neg_seen_g.nodes())


    def __add_temporal_edges_graph(self, g, timerange):
        len_t = len(timerange)
        list_timerange = list(timerange)
        nb_nodes = g.number_of_nodes()
        min_t = int(g.edata['timestamp'].min())
        max_t = int(g.edata['timestamp'].max())
        df = self.data_df[(self.data_df['t']<max_t) & (self.data_df['t']>=min_t)]

        print(f'min_t : {min_t} - max_t : {max_t}')
        print(f'Shape df : {df.shape} (original df shape : {self.data_df.shape})')

        if self.add_self_edges:
            data = np.ones(df.shape[0]*2, dtype=np.int8)
        else:
            data = np.ones(df.shape[0], dtype=np.int8)
        rows = np.array([], dtype=np.int8)
        cols = np.array([], dtype=np.int8)
        neg_rows = np.array([], dtype=np.int8)
        neg_cols = np.array([], dtype=np.int8)
        neg_data = np.array([], dtype=np.int8)

        prev_nodes = set()
        prev_nodes_neg = set()
        curr_nodes = set()
        curr_nodes_neg = set()
        prev_t = 0
        prev_t_index = 0
        prev_adj_offset = 0

        for idx, t in enumerate(df['t'].unique()):
            
            if idx % 20 == 0:
                print(f'\n IDX : {idx} ')
            t_index = list_timerange.index(t) + 1
            adj_offset = t_index * nb_nodes
            
            # Positive edges ----------------------
            df_t = df[df['t']==t].copy()

            curr_nodes = set(np.array(df_t['src']))
            curr_nodes.update(set(np.array(df_t['dest'])))

            #print(f'\nCurrent nodes: {curr_nodes}')

            src_t = np.array(df_t['src']) 
            dest_t = np.array(df_t['dest'])

            if self.add_self_edges:
                src_t_temp = src_t
                dest_t_temp = dest_t
                src_t = np.append(src_t, dest_t_temp)
                dest_t = np.append(dest_t, src_t_temp)

            # Fill COO adjacency matrix 
            rows = np.append(rows, src_t + adj_offset)
            cols = np.append(cols, dest_t + adj_offset)

            #print(f'\nROWS  : {rows}')
            #print(f'COLS  : {cols}')

            # Negative sampling ---------------------
            # All possible edges at time t
            all_edges_permutations = list(permutations(np.array(list(curr_nodes)), 2))
            #print(f'Permutations = {all_edges_permutations}')
            # Positive edges at time t
            pos_edges_list = [(u, v) for u, v in zip(src_t, dest_t)]
            #print(f'Positive edges : {pos_edges_list}')
            tries = []
            random_edges = []
            while len(neg_data) < len(pos_edges_list) and len(tries) < len(all_edges_permutations):
                # Sample random edge in all possible edges with nodes at time t
                random_idx = np.random.randint(0, len(all_edges_permutations))
                random_edge = all_edges_permutations[random_idx]
                # Keep random edge if not equal to any positive edge at time t
                if random_edge not in pos_edges_list:
                    neg_data = np.append(neg_data, 1)
                    neg_rows = np.append(neg_rows, random_edge[0])
                    neg_cols = np.append(neg_cols, random_edge[1])
                    random_edges.append(random_edge)
                tries.append(random_idx)
            curr_nodes_neg = set(neg_rows)
            curr_nodes_neg.update(set(neg_cols))

            #print(f'\nCurrent nodes NEG: {curr_nodes_neg}')
            #print(f'Negative edges : {random_edges}')
            
            # Temporal edges ---------------------- 
            # These edges are directed from previous timestep to current timestep.
            if t_index == (prev_t_index + 1):
                temporal_nodes = prev_nodes.intersection(curr_nodes)
                temporal_nodes_neg = prev_nodes_neg.intersection(curr_nodes_neg)

                #print(f'\nTemporal nodes : {temporal_nodes}')
                #print(f'Temporal nodes NEG : {temporal_nodes_neg}')
                
                # Update COO arrays
                for node in temporal_nodes:    
                    data = np.append(data, 1)
                    rows = np.append(rows, node + prev_adj_offset)
                    cols = np.append(cols, node + adj_offset)
                for node_neg in temporal_nodes_neg:
                    neg_data = np.append(neg_data, 1)
                    neg_rows = np.append(neg_rows, node + prev_adj_offset)
                    neg_cols = np.append(neg_cols, node + adj_offset)

            prev_t_index = t_index
            prev_adj_offset = adj_offset
            prev_nodes = curr_nodes
            prev_nodes_neg = curr_nodes_neg

        # Build new graphs from edges
        # A self edge is added for the last node at last timestep, in order create same-size graphs between positive and negative samples
        rows = np.append(rows, nb_nodes*len_t)
        cols = np.append(cols, nb_nodes*len_t)
        neg_rows = np.append(rows, nb_nodes*len_t)
        neg_cols = np.append(cols, nb_nodes*len_t)
        g_pdag = dgl.graph((rows, cols))
        g_pdag_neg = dgl.graph((neg_rows, neg_cols))

        if self.add_self_edges():
            g_pdag = dgl.add_self_loops(g_pdag)
            g_pdag_neg = dgl.add_self_loops(g_pdag_neg)

        print('===========================')
        print(f'Positive graph : {g_pdag}')
        print(f'Negative graph : {g_pdag_neg}')
        print('===========================')

        return g_pdag, g_pdag_neg


    def __add_temporal_edges(self):
        ''' If :math:`u(t)` and :math:`u(t+1)` both exist, adds a directed link between :math:`u(t)` and :math:`u(t+1)`. ''' 

        self.add_temporal_edges = True

        #self.g = self.__add_temporal_edges_graph2(self.g, self.trange)
        if self.is_splitted:
            #self.train_g, _ = self.__add_temporal_edges_graph(self.train_g, self.trange_train)
            self.train_pos_g, self.train_neg_g = self.__add_temporal_edges_graph(self.train_pos_g, self.trange_train)
            self.train_g = self.train_pos_g
        #    self.val_pos_g = self.__add_temporal_edges_graph(self.val_pos_g, self.trange_val)
            self.test_pos_g, self.test_neg_g = self.__add_temporal_edges_graph(self.test_pos_g, self.trange_test)
        #    self.test_pos_seen_g = self.__add_temporal_edges_graph(self.test_pos_seen_g, self.trange_test)


    def compute_features(self, feat_struct: str, add_self_edges=True, normalized=True, device='cpu'):
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
                            at which node :math:`u` is present. The size of the final adjacency matrix is :math:`(|V|*|T|, |V|*|T|)`.
                add_self_edges: bool (default=True) 
                    If True, add self-edges for each node in graph. 
                    
            Output
            -------
                Adjacency matrix as a torch tensor. '''
                            
        print('\nCompute features ...')
        print(f'    Data structure : {feat_struct}')
        self.add_self_edges = add_self_edges

        if feat_struct=='temporal_edges':

            # Create graphs with temporal edges
            if exists(f'{os.getcwd()}/data.bin'): 
                glist = list(load_graphs(f"{os.getcwd()}/data.bin")[0])
                self.train_g, self.train_pos_g, self.train_neg_g, _, _ = glist
            else:
                self.__add_temporal_edges()
                # Save results
                save_graphs("./data.bin", [self.train_g, self.train_pos_g, self.train_neg_g, self.test_pos_g, self.test_neg_g])

            # Add self edges
            if add_self_edges:
                self.train_g = dgl.add_self_loop(self.train_g)
                self.train_pos_g = dgl.add_self_loop(self.train_pos_g)

            # Compute features as coo adjacency matrix 
            adj = self.train_pos_g.adj()

        elif feat_struct=='time_tensor':

            # Build training adjacency 3d-tensor
            max_train_t = int(self.train_pos_g.edata['timestamp'].max())
            df_train = self.data_df[self.data_df['t']<=max_train_t]
            train_trange = list(np.arange(int(self.train_pos_g.edata['timestamp'].min()), int(self.train_pos_g.edata['timestamp'].max()) + 20, 20))
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

        elif feat_struct=='agg':

            # Add self-edges over time
            if add_self_edges:
                self.__add_self_edges()

            # Build averaged adjacency matrix over time
            src, dest = self.train_pos_g.edges()
            adj = coo_matrix((np.ones(len(src)), (src.numpy(), dest.numpy())))
            
            norm_mat = np.eye(adj.shape[0]) * (1 / len(self.trange_train))
            adj = torch.from_numpy(adj.dot(norm_mat))

            if normalized:
                adj = normalize_adj(adj)

        # Attach features to graph nodes
        self.train_g.ndata['feat'] = adj
        self.train_pos_g.ndata['feat'] = adj

        print(self.train_pos_g)
        print(self.test_pos_g)

        print('Done !')


    def train_test_split(self, val_size: float = 0.15, test_size: float = 0.15, neg_sampling: bool = True):

        print('\nSplitting graph ...')

        self.is_splitted = True
        self.neg_sampling = neg_sampling

        # Compute validation time and test time
        val_cut = 1 - val_size - test_size
        test_cut = 1 - test_size
        val_time, test_time = np.quantile(self.g.edata['timestamp'], [val_cut, test_cut])

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

        train_pos_g = dgl.graph((source[train_mask], dest[train_mask]), num_nodes=self.g.number_of_nodes())
        val_pos_g = dgl.graph((source[val_mask], dest[val_mask]), num_nodes=self.g.number_of_nodes())
        test_pos_g = dgl.graph((source[test_mask], dest[test_mask]), num_nodes=self.g.number_of_nodes())
        train_pos_g.edata['timestamp'] = timestamp[train_mask]
        val_pos_g.edata['timestamp'] = timestamp[val_mask]
        test_pos_g.edata['timestamp'] = timestamp[test_mask]

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
            timerange = np.arange(int(self.g.edata['timestamp'].min()), int(self.g.edata['timestamp'].max()), 20)

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
        self.trange = np.arange(int(self.g.edata['timestamp'].min()), int(self.g.edata['timestamp'].max()) + 20, 20)
        self.trange_train = self.trange[self.trange < val_time]
        self.trange_val = self.trange[(self.trange >= val_time) & (self.trange < test_time)]
        self.trange_test = self.trange[self.trange >= test_time]

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