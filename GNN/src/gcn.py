from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from layer import GCNLayer, GCNLayerNonNeighb, GCNLayer_time, GCNLayerFull
from utils import compute_auc
from overrides import overrides

class GCNModel(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCNModel, self).__init__()

    def forward(self, g, features):
        h = F.relu(self.layer1(g, features))
        h = self.layer2(g, h)
        return h

    def test(self, predictor, test_pos_g, test_neg_g, metric='auc', return_all=True):

        history = {} # useful for plots
    
        with torch.no_grad():
            pos_score = predictor(test_pos_g, self.embedding_)
            neg_score = predictor(test_neg_g, self.embedding_)
            if metric=='auc':
                auc, fpr, tpr = compute_auc(pos_score, neg_score)
            
            # Save results
            history['test_auc'] = auc
            history['test_fpr'] = fpr
            history['test_tpr'] = tpr
            
            if return_all:
                return history, pos_score, neg_score
            else:
                return history


class GCNNeighb(GCNModel):
    def __init__(self, in_feats, h_feats):
        super(GCNNeighb, self).__init__(in_feats, h_feats)
        self.layer1 = GCNLayer(in_feats, h_feats)
        self.layer2 = GCNLayer(h_feats, h_feats)

    def train(self, optimizer, pos_batches, neg_batches, emb_size, predictor, loss, device, epochs):
        
        history = defaultdict(list)

        # Initialize embedding for 1st timestep
        emb_prev = torch.rand(pos_batches[0].ndata['feat'].shape[0], emb_size, requires_grad=False)

        for epoch in range(epochs):

            # Training for each timestep (20 seconds)
            for idx, (batch_pos_g, batch_neg_g) in enumerate(zip(pos_batches, neg_batches)):
                
                # Current active nodes
                curr_nodes = torch.nonzero(batch_pos_g.ndata['feat']).squeeze().unique()
                sub_active_g = batch_pos_g.subgraph(curr_nodes).to(device) # Subgraph of active nodes (on which message passing is applied)
                batch_pos_g = batch_pos_g.to(device)
                batch_neg_g = batch_neg_g.to(device)

                # Forward on active nodes only. For other nodes, embedding at previous timestep is copied.
                emb_N_active = self.forward(sub_active_g, sub_active_g.ndata['feat'].to(torch.float32))
                emb_N = emb_prev.clone()
                for idx, node in enumerate(curr_nodes):
                    emb_N[node, :] = emb_N_active[idx, :]
                emb_prev = emb_N.clone().detach()

                emb_N = emb_N.to(device)

                pos_score = predictor(batch_pos_g, emb_N)
                neg_score = predictor(batch_neg_g, emb_N)
                loss_val = loss(pos_score, neg_score, device)

                # Save results
                history['train_loss'].append(loss_val.cpu().detach().numpy())
                history['train_emb'].append(emb_N.cpu().detach().numpy())

                # Backward
                optimizer.zero_grad() # zero the parameter gradients
                loss_val.backward()
                optimizer.step()
        
        self.embedding_ = emb_N
        self.history_train_ = history


class GCNNonNeighb(GCNModel):
    def __init__(self, in_feats, h_feats):
        super(GCNNonNeighb, self).__init__(in_feats, h_feats)
        self.layer1 = GCNLayerNonNeighb(in_feats, h_feats)
        self.layer2 = GCNLayerNonNeighb(h_feats, h_feats)

    def train(self, optimizer, pos_batches, neg_batches, emb_size, predictor, loss, device, epochs):
        
        history = defaultdict(list)

        for epoch in range(epochs):
    
            # Training for each timestep (20 seconds)
            for idx, (batch_pos_g, batch_neg_g) in enumerate(zip(pos_batches, neg_batches)):
                
                # To device
                batch_pos_g = batch_pos_g.to(device)
                batch_neg_g = batch_neg_g.to(device)

                # Forward on active nodes only. For other nodes, embedding at previous timestep is copied.
                emb_NN = self.forward(batch_pos_g, batch_pos_g.ndata['feat'].to(torch.float32))
                emb_NN = emb_NN.to(device)
                pos_score = predictor(batch_pos_g, emb_NN)
                neg_score = predictor(batch_neg_g, emb_NN)
                loss_val = loss(pos_score, neg_score, device)
                        
                # Save results
                history['train_loss'].append(loss_val.cpu().detach().numpy())
                history['train_emb'].append(emb_NN.cpu().detach().numpy())
                
                # Backward
                optimizer.zero_grad() # zero the parameter gradients
                loss_val.backward()
                optimizer.step()
        
        self.embedding_ = emb_NN
        self.history_train_ = history

        
class GCNModel_time(GCNModel):
    def __init__(self, in_feats, h_feats, time_dim):
        super(GCNModel_time, self).__init__(in_feats, h_feats)
        self.layer1 = GCNLayer_time(in_feats, time_dim, h_feats)
        self.layer2 = GCNLayer_time(h_feats, 1, h_feats)
        self.layer3 = GCNLayer_time(h_feats, 1, h_feats)
        
    @overrides
    def forward(self, g, features):
        h = F.relu(self.layer1(g, features))
        h = F.relu(self.layer2(g, h))
        h = self.layer3(g, h)
        return h  

    def train(self, optimizer, train_g, train_pos_g, train_neg_g, predictor, loss, device, epochs):
        
        history = defaultdict(list) # Useful for plots
        
        for epoch in range(epochs):
            
            # To device
            train_g = train_g.to(device)
            train_pos_g = train_pos_g.to(device)
            train_neg_g = train_neg_g.to(device)

            # forward
            h = self.forward(train_g, train_g.ndata['feat'].to(torch.float32)).cpu()
            pos_score = predictor(train_pos_g, h.to(device))
            neg_score = predictor(train_neg_g, h.to(device))
            loss_val = loss(pos_score, neg_score, device)
            
            #Save results
            history['train_loss'].append(loss_val.cpu().detach().numpy())
            
            # backward
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if epoch%100==0:
                print(f'In epoch {epoch}, loss: {loss_val:.4f}')

        self.embedding_ = h
        self.history_train_ = history


class GCNModelFull(GCNModel):
    def __init__(self, in_feats, h_feats):
        super(GCNModelFull, self).__init__(in_feats, h_feats)
        self.layer1 = GCNLayerFull(h_feats, h_feats)
        self.layer2 = GCNLayerFull(h_feats, h_feats)

    def train(self, optimizer, pos_batches, neg_batches, emb_size, predictor, loss, device, epochs, \
                emb_prev, emb_neighbors, emb_nneighbors, \
                alpha=0.5, beta=0.25, gamma=0.25):
        
        history = defaultdict(list)
        #history_emb_tot = []

        for epoch in range(epochs):
    
            # Training for each timestep (20 seconds)
            for idx, (batch_pos_g, batch_neg_g) in enumerate(zip(pos_batches, neg_batches)):

                # Saved embeddings
                h_Neighb = torch.from_numpy(emb_neighbors[idx]).requires_grad_(False)
                h_NNeighb = torch.from_numpy(emb_nneighbors[idx]).requires_grad_(False)

                # To device
                batch_pos_g = batch_pos_g.to(device)
                batch_neg_g = batch_neg_g.to(device)
                h_Neighb = h_Neighb.to(device)
                h_NNeighb = h_NNeighb.to(device)
                emb_prev = emb_prev.to(device)

                # forward
                emb_mem_active = self.forward(batch_pos_g, emb_prev)
                emb_tot = alpha*h_Neighb + beta*emb_mem_active + gamma*h_NNeighb
                emb_tot = emb_tot.to(device)

                pos_score = predictor(batch_pos_g, emb_tot)
                neg_score = predictor(batch_neg_g, emb_tot)
                loss_val = loss(pos_score, neg_score, device)

                # Save results
                history['train_loss'].append(loss_val.cpu().detach().numpy())
                history['train_emb'].append(emb_tot.cpu().detach().numpy())
                
                # Backward
                optimizer.zero_grad() # zero the parameter gradients
                loss_val.backward()
                optimizer.step()
            
            if epoch==(epochs-1):
                emb_prev = emb_tot.clone().detach()
                #history_emb_tot.append(emb_prev)
        
        self.embedding_ = emb_tot
        self.history_train_ = history
        #self.history_emb_ = history_emb_tot
        self.history_emb_ = emb_prev