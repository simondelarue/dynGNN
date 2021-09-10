import numpy as np
import dgl
import torch

def temporal_sampler(g, batch_size, timestep):
    ''' Returns a list of subgraph according to desired batch size. '''
    
    batches = []
    indexes = [] # returns list of index with 1 if batch-graph was returned, else 0
    
    batch_period = timestep * batch_size
    timerange = np.arange(int(g.edata['timestamp'].min()), int(g.edata['timestamp'].max()), batch_period)
    eids = np.arange(g.number_of_edges())
    
    for period in timerange:
    
        # Edges to remove
        rm_eids = eids[torch.logical_not(torch.logical_and(g.edata['timestamp'] >= period, 
                                                           g.edata['timestamp'] < (period + batch_period)))]
        
        batch_g = dgl.remove_edges(g, rm_eids) # also remove the feature attached to the edge
        
        # Later, use indexes to consider graph batch only if edges exist inside
        batches.append(batch_g)
        
        if batch_g.number_of_edges() != 0:
            indexes.append(True)
        else:
            indexes.append(False)
        
    return batches, indexes