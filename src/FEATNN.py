from torch import nn
from ProtExtractor import ProteinExtractor as ProteinExtractor
from CompExtractor import CompExtractor
from AffinityPrediction import AffinityPredModule


class FEATNN(nn.Module):
    def __init__(self, init_atom_features, init_bond_features, param):
        super(FEATNN, self).__init__()

        self.init_atom_features = init_atom_features
        self.init_bond_features = init_bond_features
        self.compextr = CompExtractor(param['comp'], self.init_atom_features, self.init_bond_features)
        self.protextr = ProteinExtractor(param['prot'])

        self.affinitypred = AffinityPredModule(param['affinity_pred'])

    def forward(self, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, distance_mat, torsion_mat, dist_mask):
        batch_size = vertex.size(0)
        comp_features, master_features = self.compextr(batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask)
        protein_features, pair_repr = self.protextr(sequence, seq_mask, distance_mat, torsion_mat, dist_mask)

        affinity_pred = self.affinitypred(batch_size, comp_features, protein_features, master_features, vertex_mask, seq_mask)

        return affinity_pred
