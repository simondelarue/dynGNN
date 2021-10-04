import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from overrides import overrides
from utils import sample_non_neighbors, sample_neighbors

class GCNLayer(nn.Module):
    def __init__(self, features_in, features_out):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(features_in, features_out)
        
    def forward(self, g, features):
        with g.local_scope():
            g.ndata['h'] = features
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.sum('m', 'h_N'))
            h_N = F.normalize(g.ndata['h'] + g.ndata['h_N'], dim=1)
            return self.linear(h_N)


class GCNLayerTime(GCNLayer):
    def __init__(self, features_in, time_dimension, features_out):
        super(GCNLayerTime, self).__init__(features_in, features_out)
        self.linear_time = nn.Linear(features_in*time_dimension, features_out)
    
    @overrides
    def forward(self, g, features):
        with g.local_scope():
            g.ndata['h'] = features
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.sum('m', 'h_N'))
            h_N = torch.flatten(g.ndata['h_N'], start_dim=1)
            h_N = F.normalize(h_N, dim=1)
            return self.linear_time(h_N)


class GCNLayerNonNeighb(GCNLayer):
    def __init__(self, features_in, features_out):
        super(GCNLayerNonNeighb, self).__init__(features_in, features_out)

    @overrides
    def forward(self, g, features):

        # Current active nodes
        curr_nodes = torch.nonzero(g.ndata['feat']).squeeze().unique()
        
        # All nodes set 
        all_nodes_set = set(g.nodes().cpu().detach().numpy())
        
        # Initialize Non-neighbors embedding Tensor
        h_NN = features.clone()

        # Update embedding (for active nodes at timestep only) using non-neighbors features
        for node in curr_nodes:
            non_neighbors_nodes = sample_non_neighbors(g, node.cpu(), all_nodes_set)
            h_NN_tmp = torch.sum(features[non_neighbors_nodes], dim=0)
            h_NN[node] = features[node] + h_NN_tmp
        
        # Normalize result
        h_NN_norm = F.normalize(h_NN, dim=1)
        
        return self.linear(h_NN_norm)


class GCNLayerFull(GCNLayer):
    def __init__(self, features_in, features_out):
        super(GCNLayerFull, self).__init__(features_in, features_out)
        self.linear = nn.Linear(features_in, features_out)
        
    @overrides
    def forward(self, g, features):
        
        # Current active nodes
        curr_nodes = torch.nonzero(g.ndata['feat']).squeeze().unique()
        
        # Initialize Non-neighbors embedding Tensor
        h_N = features.clone()
        
        # Update only current node's embedding
        for node in curr_nodes:
            neighbors_nodes = sample_neighbors(g, node)
            h_N_tmp = torch.sum(features[neighbors_nodes, :], dim=0)
            h_N[node, :] = features[node, :] + h_N_tmp
            
        # Normalize result
        h_N_norm = F.normalize(h_N, dim=1)
        
        return self.linear(h_N_norm)