import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import numpy as np
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.distances.lp_distance import LpDistance
from pytorch_metric_learning.reducers import MeanReducer
from ms_extended import MultiSimilarityMinerExtended
import math


class TransE(nn.Module):
    def __init__(self, margin=1.0):
        super(TransE, self).__init__()
        self.margin = margin

    def forward(self, cui_0, cui_1, cui_2, re):
        pos = cui_0 + re - cui_1
        neg = cui_0 + re - cui_2
        return torch.mean(F.relu(self.margin + torch.norm(pos, p=2, dim=1) - torch.norm(neg, p=2, dim=1)))

class TransE_batch(TransE):
    def __init__(self, margin=1.0):
        super(TransE_batch, self).__init__()
        self.margin = margin

    def forward(self, cui_0, cui_1, re):
        t = cui_0 + re
        pos = (t - cui_1).unsqueeze(1)
        neg = t.unsqueeze(1) - cui_1.unsqueeze(0)
        return torch.mean(F.relu(self.margin + torch.norm(pos, p=2, dim=-1) - torch.norm(neg, p=2, dim=-1)))


class TransE_MS(nn.Module):
    def __init__(self, factor=10):
        super(TransE_MS, self).__init__()
        self.dis = LpDistance(normalize_embeddings=False, p=2)
        self.factor = factor
        self.loss_fn = losses.MultiSimilarityLoss(alpha=2 / factor,
                                                  beta=50 / factor,
                                                  base=28,
                                                  distance=self.dis)
        self.miner = MultiSimilarityMinerExtended(epsilon=0.1, distance=self.dis)

    def forward(self, cui_0, cui_1, re, cui_label_0, cui_label_1):
        t = cui_0 + re
        a1, p, a2, n = self.miner(t, cui_label_1, cui_1, cui_label_1)
        mat = self.dis(t, cui_1)
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        loss = self.loss_fn._compute_loss(mat, pos_mask, neg_mask)
        return torch.mean(loss['loss']['losses']) * 1 / self.factor

class CosineDistMult_MS(nn.Module):
    def __init__(self, rel_count, hidden_dim=768):
        super(CosineDistMult_MS, self).__init__()
        self.dis = CosineSimilarity()
        self.loss_fn = losses.MultiSimilarityLoss(alpha=2, beta=50)
        self.miner = MultiSimilarityMinerExtended(epsilon=0.1)
        self.rel_count = rel_count
        self.hidden_dim = hidden_dim
        self.transform = torch.nn.Parameter(torch.FloatTensor(rel_count, hidden_dim, hidden_dim))
        stdv = 1. / math.sqrt(hidden_dim)
        self.transform.data.uniform_(-stdv, stdv)

    def forward(self, cui_0, cui_1, re_id, cui_label_0, cui_label_1):
        m_r = self.transform[re_id] # batch * d * d
        t = torch.bmm(m_r, cui_0.unsqueeze(-1)).squeeze(-1)
        
        a1, p, a2, n = self.miner(t, cui_label_1, cui_1, cui_label_1)
        mat = self.dis(t, cui_1)
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        loss = self.loss_fn._compute_loss(mat, pos_mask, neg_mask)
        return torch.mean(loss['loss']['losses']) * 1

