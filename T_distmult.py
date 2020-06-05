# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import Dataset

class T_distmult(torch.nn.Module):
    def __init__(self, dataset, params):
        super(T_distmult, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs = nn.Embedding(dataset.numEnt(), params.emb_dim).cuda()
        self.rel_embs = nn.Embedding(dataset.numRel(), params.emb_dim).cuda()
        self.tim_embs = nn.Embedding(dataset.numTime(), params.t_emb_dim).cuda()
    

        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        nn.init.xavier_uniform_(self.tim_embs.weight)

    def getEmbeddings(self, heads, rels, tails, dates, intervals = None):

        h_embs1 = self.ent_embs(heads)
        r_embs1 = self.rel_embs(rels)
        t_embs1 = self.ent_embs(tails)
        T_embs1 = self.tim_embs(dates)
        T_embs2 = torch.ones(dates.__len__(), self.params.s_emb_dim).cuda()
        T_embs1 = torch.cat((T_embs1, T_embs2), 1)

        return h_embs1, r_embs1, t_embs1, T_embs1
    
    def forward(self, heads, rels, tails,dates):
        h_embs1, r_embs1, t_embs1, T_embs1= self.getEmbeddings(heads, rels, tails,dates)
        scores = (h_embs1 * r_embs1) * t_embs1*T_embs1
        #scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores
    
#    def forward(self, heads, rels, tails,dates):
#        #h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 ,T_embs1,T_embs2= self.getEmbeddings(heads, rels, tails,dates)
#        h, r, t, T= self.getEmbeddings(heads, rels, tails,dates)
#        t = t.view(-1, self.params.emb_dim, 1)
#        r = r.view(-1, self.params.emb_dim, self.params.emb_dim)
#        tr = torch.matmul(r, t)
#        tr = tr.view(-1, self.params.emb_dim)
#        scores=-h*tr*T
#        #scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
#        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
#        scores = torch.sum(scores, dim=1)
#        return scores
