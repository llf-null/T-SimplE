# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import Dataset

class T_complex(torch.nn.Module):
    def __init__(self, dataset, params):
        super(T_complex, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs_h = nn.Embedding(dataset.numEnt(), params.emb_dim)
        self.ent_embs_t = nn.Embedding(dataset.numEnt(), params.emb_dim)
        self.rel_embs_f = nn.Embedding(dataset.numRel(), params.emb_dim)
        self.rel_embs_i = nn.Embedding(dataset.numRel(), params.emb_dim)
        self.tim_embs_f = nn.Embedding(dataset.numTime(), params.t_emb_dim)
    
        
        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)
        nn.init.xavier_uniform_(self.tim_embs_f.weight)

    def getEmbeddings(self, heads, rels, tails, dates, intervals = None):

        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        T_embs1 = self.tim_embs_f(dates)
        T_embs2 = torch.ones(dates.__len__(), self.params.s_emb_dim).cuda()
        T_embs1 = torch.cat((T_embs1, T_embs2), 1)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)
        
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2,T_embs1
    
    def forward(self, heads, rels, tails,dates):
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 ,T_embs1= self.getEmbeddings(heads, rels, tails,dates)
        scores = h_embs1*t_embs1*r_embs1*T_embs1+h_embs2*t_embs2*r_embs1*T_embs1+h_embs1*t_embs2*r_embs2*T_embs1-h_embs2*t_embs1*r_embs2*T_embs1
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores
        

