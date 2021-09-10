import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class DotPredictor(nn.Module):
    def forward(self, g, mem):
        with g.local_scope():
            g.ndata['h'] = mem
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'].squeeze(dim=1)