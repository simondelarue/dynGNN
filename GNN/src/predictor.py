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


def udf_u_cos_v(edges):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return {'cos': cos(edges.src['h'], edges.dst['h'])}


class CosinePredictor(nn.Module):
    def forward(self, g, h):
        """
        graph : graph with edges connecting pairs of nodes
        h : hidden state of every node
        """
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(udf_u_cos_v)
            return g.edata['cos']#.squeeze(dim=1)