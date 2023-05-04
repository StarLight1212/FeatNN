import torch
from torch import nn


class ProteinEmbedding(nn.Module):
    def __init__(self, param):
        super(ProteinEmbedding, self).__init__()
        self.amino_vocab = param['Amino_vocab']
        self.embed_size = param['embed_size']
        self.dist_vocab = param['Dist_vocab']
        self.hidden_size = param['hidden_size']
        self.kernel = param['kernel_size']
        self.torsion_size = param['torsion_size']
        self.distance_embedding = nn.Embedding(self.dist_vocab, self.embed_size, padding_idx=0)
        self.word_embedding = nn.Embedding(self.amino_vocab, self.embed_size, padding_idx=0)
        self.seq_CNN = nn.Conv1d(self.embed_size, self.hidden_size, \
                                 kernel_size=self.kernel, padding=int((self.kernel-1)/2))
        self.tor_CNN = nn.Conv1d(self.torsion_size, self.embed_size, \
                                 kernel_size=self.kernel, padding=int((self.kernel-1)/2))
        self.tor2seq = nn.Conv1d(self.embed_size, self.hidden_size, \
                                 kernel_size=self.kernel, padding=int((self.kernel-1)/2))
        self.tor_gate = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, sequence, seq_mask, distance_mat, torsion_mat, dis_mask):
        # bz, seq, pre_embed
        seq_embds = self.word_embedding(sequence)
        # bz, seq, embed
        seq_features = seq_embds*seq_mask.unsqueeze(2)
        # bz, seq, hidden_size
        dis_embds = self.distance_embedding(distance_mat)*dis_mask.unsqueeze(3)
        tor_vec = nn.GELU()(self.tor_CNN(torsion_mat.transpose(1, 2)))
        tor_vec = self.tor2seq(tor_vec).transpose(1, 2)
        gate = nn.Sigmoid()(self.tor_gate(tor_vec))

        seq_features = seq_features.transpose(1, 2)
        seq_features = nn.GELU()(self.seq_CNN(seq_features))
        seq_features = seq_features.transpose(1, 2)
        seq_features = gate*tor_vec + (1-gate)*seq_features
        return seq_features, dis_embds


class DeepSeparseCNN(nn.Module):
    def __init__(self, in_ch, out_ch, param):
        super(DeepSeparseCNN, self).__init__()
        self.kernel = param['kernel_size']
        self.depth_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=self.kernel,
            stride=1,
            padding=int((self.kernel-1)/2),
            groups=in_ch
        )
        self.point_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class CNN_block(nn.Module):
    def __init__(self, param):
        super(CNN_block, self).__init__()
        self.hidden_size = param['hidden_size']
        self.kernel = param['kernel_size']
        self.num_attention_head = param['num_attention_head']
        self.attention_head_size = int(self.hidden_size/self.num_attention_head)
        self.C = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.c1 = nn.Conv1d(self.attention_head_size, self.hidden_size, \
                            kernel_size=self.kernel, padding=int((self.kernel-1)/2))
        self.c2 = nn.Conv1d(self.attention_head_size, self.hidden_size, \
                            kernel_size=self.kernel, padding=int((self.kernel-1)/2))
        self.c3 = nn.Conv1d(self.attention_head_size, self.hidden_size, \
                            kernel_size=self.kernel, padding=int((self.kernel-1)/2))
        self.c4 = nn.Conv1d(self.attention_head_size, self.hidden_size, \
                            kernel_size=self.kernel, padding=int((self.kernel-1)/2))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_head,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, x):
        x0 = self.C(x.transpose(1, 2)).transpose(1, 2)
        x = self.transpose_for_scores(x)
        x = x.transpose(1, 3)
        x = self.c1(x[:, :, 0, :]).transpose(1, 2) + self.c2(x[:, :, 1, :]).transpose(1, 2) + self.c3(
            x[:, :, 2, :]).transpose(1, 2) + self.c4(x[:, :, 3, :]).transpose(1, 2)
        x = x+x0
        return x


class EvoAttention(nn.Module):
    def __init__(self, param):
        super(EvoAttention, self).__init__()
        self.embed = param['embed_size']
        self.hidden_size = param['hidden_size']
        self.num_attention_head = param['num_attention_head']
        self.attention_head_size = int(self.hidden_size/self.num_attention_head)
        self.seq_mixture_key1 = DeepSeparseCNN(self.embed, self.hidden_size, param)
        self.seq_mixture_key2 = DeepSeparseCNN(self.embed, self.hidden_size, param)
        self.g1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.g2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.seq2struct = DeepSeparseCNN(self.hidden_size, self.hidden_size, param)
        self.GRU1 = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.GRU2 = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.mapping = DeepSeparseCNN(self.hidden_size, self.embed, param)
        self.seq_CNN = CNN_block(param)
        self.struct_CNN = CNN_block(param)
        self.GRU3 = nn.GRUCell(self.hidden_size, self.hidden_size)

    def self_outer_sum(self, seq_eins, seq_zwei):
        """
        Parameters
        ----------
        seq_eins: batch_size, seq, embed_size
        seq_zwei: batch_size, seq, embed_size

        Returns: batch_size, seq, seq, embed_size
        -------

        """
        return seq_eins.unsqueeze(1) + seq_zwei.unsqueeze(2)

    def forward(self, prot_features, pair_repr, seq_mask, dist_mask):
        batch_size, seq_len, _ = prot_features.shape
        seq_mask = seq_mask.unsqueeze(2)
        # Seq update
        prot_features = self.seq_CNN(prot_features)
        prot_features = nn.GELU()(prot_features)
        # Struct Update

        # struct2seq
        key1 = nn.GELU()(self.seq_mixture_key1((torch.sum(pair_repr, dim=1)/seq_len).transpose(1,2))).transpose(1,2)
        key2 = nn.GELU()(self.seq_mixture_key2((torch.sum(pair_repr, dim=-2)/seq_len).transpose(1,2))).transpose(1,2)
        key = self.GRU3(key1.reshape(-1, 128), key2.reshape(-1, 128)).view(batch_size, -1, 128)
        struct = self.struct_CNN(key)

        # seq2struct
        seq2struct_vec = nn.GELU()(self.seq2struct(prot_features.transpose(1, 2)).transpose(1, 2))
        gate2 = nn.Sigmoid()(self.g2(seq2struct_vec))
        struct_vector = gate2*seq2struct_vec+(1-gate2)*struct
        struct_vector = nn.GELU()(self.GRU2(struct_vector.reshape(-1, 128), struct.reshape(-1, 128)).view(batch_size, -1, 128))
        str_map_seq = self.mapping(struct_vector.transpose(1, 2)).transpose(1, 2)
        struct_mat = self.self_outer_sum(str_map_seq, str_map_seq)*(dist_mask.unsqueeze(3))

        gating1 = nn.Sigmoid()(self.g1(key))
        prot_vector = gating1*struct + (1-gating1)*prot_features
        prot_vector = nn.GELU()(self.GRU1(prot_vector.reshape(-1, 128), prot_features.reshape(-1, 128)).view(batch_size, -1, 128))
        prot_vector = prot_vector*seq_mask
        return prot_vector, struct_mat


class ProteinEncoder(nn.Module):
    def __init__(self, param):
        super(ProteinEncoder, self).__init__()
        # self.dropout_prob = param['Dropout']
        # self.hidden_size = param['hidden_size']
        self.evo_attention = EvoAttention(param)

    def forward(self, prot_features, pair_repr, seq_mask, dist_mask):
        prot_features, pair_repr = self.evo_attention(prot_features, pair_repr, seq_mask, dist_mask)
        return prot_features, pair_repr


class ProteinExtractor(nn.Module):
    def __init__(self, param):
        super(ProteinExtractor, self).__init__()
        self.prot_embedding = ProteinEmbedding(param)
        self.encoder_layers = param['Encoder_layers_num']
        self.prot_encoders = nn.ModuleList([ProteinEncoder(param) for _ in range(self.encoder_layers)])

    def forward(self, sequence, seq_mask, distance_mat, torsion_mat, dist_mask):
        batch_size = sequence.size(0)
        prot_features, pair_repr = self.prot_embedding(sequence, seq_mask, distance_mat, torsion_mat, dist_mask)
        prot_pair_features = [(prot_features, pair_repr)]
        for i, prot_encoder in enumerate(self.prot_encoders):
            prot_features, pair_repr = prot_pair_features[-1]
            prot_feats, pair_repr = prot_encoder(prot_features, pair_repr, seq_mask, dist_mask)
            prot_pair_features.append((prot_feats, pair_repr))
        prot_features, pair_repr = prot_pair_features[-1]

        return prot_features, pair_repr

