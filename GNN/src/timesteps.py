import dgl
import random

def step_linkpred_preprocessing(g , timerange, negative_sampling: bool = True):
    ''' Cut down given temporal graph into subgraphs, according to timesteps. Performs negative sampling on each
        snapshot if needed.
        
        Parameters
        -----------
            g : dgl Graph
            timerange : np.ndarray
                Array of timesteps
            negative_sampling : bool (default=True)
                If True, performs negative sampling on each snapshot, e.g for each graph at each timestep. 
                
        Output
        -------
            val_pos_g_list, val_neg_g_list : Two lists of dgl Graphs (of length equals to number of timesteps), containing positive, 
            resp. negative sampling graphs for each timestep. ''' 

    # Results lists
    val_pos_g_list = []
    val_neg_g_list = []

    def edges_with_feature_t(edges):
        # Whether an edge has a timestamp equals to t
        return (edges.data['timestamp'] == val_t)

    for t in timerange:
        # -------- Positive edges ---------
        global val_t
        val_t = t
        eids = g.filter_edges(edges_with_feature_t) # Filters edges for each timestep
        val_pos_g = dgl.edge_subgraph(g, eids, preserve_nodes=True)
        val_pos_g_list.append(val_pos_g)
        src_t, dest_t = val_pos_g.edges()

        # -------- Negative edges ---------
        val_neg_g = step_linkpred_neg_sampling(src_t, dest_t, val_pos_g.number_of_nodes(), k=3)
        val_neg_g_list.append(val_neg_g)

    return val_pos_g_list, val_neg_g_list


def step_linkpred_neg_sampling(src_nodes, dest_nodes, n, k=3):
    ''' Performs negative sampling on the basis of positive links from a graph and returns negative sampled graph. 
        If the total number of nodes in positive graph is <= 2, the negative sampled graph is empty.
        Note : If after :math:`k` tries, no negative edge is found, the total number of negative edges can be smaller
        than the number of positive edges. 
        
        Parameters
        -----------
            src_nodes, dest_nodes : np.ndarray, np.ndarray
                Contains source nodes, resp. destination nodes, in positive graph
            n : int
                Number of nodes in output graph. 
            k : int (default=3)
                Number of iterations to perform in order to find a negative edge, given a specific node. 
                
        Output
        -------
            g : dgl.Graph
                Negative sampled graph. '''

    # Positive edges as a list
    pos_edge_list = [(u, v) for u, v in zip(src_nodes, dest_nodes)]
    pos_nodes = set(src_nodes).union(set(dest_nodes))

    rows_neg = []
    cols_neg = []

    # If only 2 nodes in graph at time t, we cannot create negative edges
    if len(pos_nodes) > 2:

        for src_node in src_nodes:
            pos_nodes_list = list(pos_nodes)
            pos_nodes_list.remove(src_node) # Self edge is not allowed
        
            # Performs k iterations to find negative edge, given one of its nodes
            for i in range(k):
                neg_edge = (src_node, random.choice(pos_nodes_list)) 
                if neg_edge not in pos_edge_list:
                    rows_neg.append(neg_edge[0])
                    cols_neg.append(neg_edge[1])
                    break

    # Negative sampled graph
    val_neg_g = dgl.graph((rows_neg, cols_neg), num_nodes=n)
    
    return val_neg_g