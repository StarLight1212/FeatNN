import torch
from torch import nn
import torch.nn.functional as F


class AffinityPredModule(nn.Module):
    def __init__(self, param):
        super(AffinityPredModule, self).__init__()
        self.hidden_size = param['hidden_size']
        self.kernel = param['kernel_size']
        """Affinity Prediction Module"""
        # Feature Extraction of Prot, Comp and Master
        self.master_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.p_final = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel, padding=int((self.kernel-1)/2))

        # Output layer
        self.aff = nn.Linear(self.hidden_size * self.hidden_size * 2, 1)

    def forward(self, batch_size, comp_feature, prot_feature, master_feature, vertex_mask, seq_mask):

        comp_feature = F.gelu(self.c_final(comp_feature))
        prot_feature = F.gelu(self.p_final(prot_feature.transpose(1,2))).transpose(1,2)
        super_feature = F.gelu(self.master_final(master_feature.view(batch_size, -1)))
        vertex_mask = vertex_mask.view(batch_size, -1, 1)
        seq_mask = seq_mask.view(batch_size, -1, 1)

        c0 = torch.sum(comp_feature * vertex_mask, dim=1) / torch.sum(vertex_mask, dim=1)
        p0 = torch.sum(prot_feature * seq_mask, dim=1) / torch.sum(seq_mask, dim=1)

        cf = torch.cat([c0.view(batch_size, -1), super_feature.view(batch_size, -1)], dim=1)
        # aggr_info (bz, hidden_size*hidden_size*2)
        aggr_info = F.gelu(
            torch.matmul(cf.view(batch_size, -1, 1), p0.view(batch_size, 1, -1)).view(batch_size, -1))
        # affinity_pred (bz, 1)
        affinity_pred = self.aff(aggr_info)

        return affinity_pred