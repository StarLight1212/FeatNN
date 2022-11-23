from torch import nn
import torch
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, param, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        self.bond_fdim = param['bond_fdim']
        if self.variant:
            self.in_feature = 2*in_features
        else:
            self.in_feature = in_features

        self.out_feature = out_features
        self.residual = residual

        self.label_U1 = nn.Linear(self.out_feature + self.bond_fdim, self.out_feature)
        self.label_U2 = nn.Linear(self.out_feature * 2, self.out_feature)
        self.feature_updating = nn.Linear(self.in_feature, self.out_feature)

    def forward(self, vertex_features, atom_adj, bond_adj, h0, lamda, alpha, l, edge_initial, vertex_mask, nbs_mask):
        batch_size = vertex_mask.size(0)
        n_vertex = vertex_mask.size(1)
        n_nbs = nbs_mask.size(2)
        nbs_mask = nbs_mask.view(batch_size, n_vertex, n_nbs, 1)
        # Atom neighbor features
        # (batch_size, atom_num, neighbor_dim, hidden_size)
        vertex_neighbor = torch.index_select(vertex_features.view(-1, self.out_feature), 0, atom_adj).view(batch_size, \
                                                                                                           n_vertex, n_nbs, self.out_feature)
        # Edge neighbor features
        # (batch_size, atom_num, neighbor_dim, bond_size)
        edge_neighbor = torch.index_select(edge_initial.view(-1, self.bond_fdim), 0, bond_adj).view(batch_size,\
                                                                                                    n_vertex, n_nbs, self.bond_fdim)
        # (batch_size, atom_num, neighbor_dim, hidden_size+bond_dim(128+6))
        l_neighbor = torch.cat((vertex_neighbor, edge_neighbor), -1)
        # Mapping
        # (batch_size, atom_num, neighbor_dim, hidden_size)
        neighbor_label = F.gelu(self.label_U1(l_neighbor))
        # Mask non-atom part
        # (batch_size, atom_num, neighbor_dim, hidden_size）->(batch_size, atom_num, hidden_size)
        neighbor_label = torch.sum(neighbor_label * nbs_mask, dim=-2)
        # cat neighbor features with vertex features
        # (batch_size, atom_num, hidden_size*2)
        hi = torch.cat((neighbor_label, vertex_features), 2)
        # Mapping
        # (batch_size, atom_num, hidden_size)
        hi = self.label_U2(hi)

        theta = math.log(lamda / l + 1)

        if self.variant:
            # cat h0 and hi in atom dimension
            # hi: (batch_size, atom_num, hidden_size)
            # h0 (batch_size, atom_num, hidden_size)
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * self.feature_updating(support) + (1 - theta) * r

        if self.residual:
            output = output + vertex_features
        return output


class CompExtractor(nn.Module):
    def __init__(self, param, init_atom_features, init_bond_features):
        super(CompExtractor, self).__init__()
        self.dropout = param['dropout']
        self.head_num = param['k_head']
        self.alpha = param['alpha']
        self.lamda = param['lamda']
        self.GCN_depth = param['GCN_depth']
        self.atom_fdim = param['atom_fdim']
        self.bond_fdim = param['bond_fdim']
        self.max_nb = param['max_nb']
        self.hidden_size1 = param['hidden_size']
        self.init_atom_features = init_atom_features
        self.init_bond_features = init_bond_features
        self.vertex_embedding = nn.Linear(self.atom_fdim, self.hidden_size1)
        self.GCN_layer = nn.ModuleList([GraphConvolution(self.hidden_size1, self.hidden_size1, param, \
                                                         variant=False, residual=False) for _ in range(self.GCN_depth)])
        # k_head attention layer
        self.W_vertex_main = nn.ModuleList(
            [nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.head_num)]) for _ in
             range(self.GCN_depth)])
        self.W_vertex_master = nn.ModuleList(
            [nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.head_num)]) for _ in
             range(self.GCN_depth)])
        self.W_main = nn.ModuleList(
            [nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.head_num)]) for _ in
             range(self.GCN_depth)])
        self.W_bmm = nn.ModuleList(
            [nn.ModuleList([nn.Linear(self.hidden_size1, 1) for _ in range(self.head_num)]) for _ in
             range(self.GCN_depth)])
        self.khead_cat_to_master = nn.ModuleList([nn.Linear(self.hidden_size1*self.head_num, self.hidden_size1) \
                                       for _ in range(self.GCN_depth)])

        self.W_master = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GCN_depth)])

        self.W_master_to_main = nn.ModuleList(
            [nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GCN_depth)])

        self.W_zm1 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GCN_depth)])
        self.W_zm2 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GCN_depth)])
        self.W_zs1 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GCN_depth)])
        self.W_zs2 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GCN_depth)])
        self.GRU_main = nn.GRUCell(self.hidden_size1, self.hidden_size1)
        self.GRU_master = nn.GRUCell(self.hidden_size1, self.hidden_size1)

    def mask_softmax(self, a, mask, dim=-1):
        batch_size = a.size(0)
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
        return a_softmax

    def forward(self, batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask):
        n_vertex = vertex.size(1)
        n_nbs = atom_adj.size(-1)
        # Initial feature from atom & bond dict
        vertex_initial = torch.index_select(self.init_atom_features, 0, vertex.view(-1))
        vertex_initial = vertex_initial.view(batch_size, -1, self.atom_fdim)
        edge_initial = torch.index_select(self.init_bond_features, 0, edge.view(-1))
        edge_initial = edge_initial.view(batch_size, -1, self.bond_fdim)
        # Masking the wrong zero-index feature

        vertex_initial = vertex_initial * (vertex_mask.view(batch_size, -1, 1))

        # edge_initial = edge_initial * edge_mask.view(batch_size, -1, 1)

        vertex_feature = F.gelu(self.vertex_embedding(vertex_initial))
        vertex_feature = vertex_feature.view(batch_size, -1, self.hidden_size1)
        vertex_feature_h0 = vertex_feature[:]
        vertex_mask = vertex_mask.view(batch_size, -1, 1)
        master_feature = torch.sum(vertex_feature * vertex_mask, dim=1, keepdim=True)

        # Deep graph convolution neural networks
        for i, GraphCovNN in enumerate(self.GCN_layer):
            for k in range(self.head_num):
                main_vertex = torch.tanh(self.W_vertex_main[i][k](vertex_feature))
                vertex = self.W_bmm[i][k](main_vertex*master_feature)
                attention_score = self.mask_softmax(vertex.view(batch_size, -1), vertex_mask.view(batch_size, -1)).view(batch_size, -1, 1)

                k_head_main_to_master = torch.bmm(attention_score.transpose(1, 2), self.W_main[i][k](vertex_feature))
                if k == 0:
                    m_main_to_master = k_head_main_to_master
                else:
                    # concat k-head
                    m_main_to_master = torch.cat([m_main_to_master, k_head_main_to_master], dim=-1)

            main_to_master = torch.tanh(self.khead_cat_to_master[i](m_main_to_master))

            layer_inner = F.dropout(vertex_feature, self.dropout, training=self.training)
            self_vertex = GraphCovNN(layer_inner, atom_adj, bond_adj, vertex_feature_h0, self.lamda, self.alpha, i+1, edge_initial, vertex_mask, nbs_mask)

            master_to_main = F.gelu(self.W_master_to_main[i](master_feature))
            master_self = F.gelu(self.W_master[i](master_feature))

            # warp gate and GRU for update main node features, use main_self and super_to_main
            z_main = torch.sigmoid(self.W_zm1[i](self_vertex) + self.W_zm2[i](master_to_main))
            hidden_main = (1 - z_main) * self_vertex + z_main * master_to_main
            vertex_feature = self.GRU_main(hidden_main.view(-1, self.hidden_size1),
                                           vertex_feature.view(-1, self.hidden_size1))
            vertex_feature = vertex_feature.view(batch_size, n_vertex, self.hidden_size1)

            # warp gate and GRU for update super node features
            z_master = torch.sigmoid(self.W_zs1[i](master_self) + self.W_zs2[i](main_to_master))
            hidden_super = (1 - z_master) * master_self + z_master * main_to_master
            master_feature = self.GRU_master(hidden_super.view(batch_size, self.hidden_size1),
                                           master_feature.view(batch_size, self.hidden_size1))
            master_feature = master_feature.view(batch_size, 1, self.hidden_size1)

        return vertex_feature, master_feature
