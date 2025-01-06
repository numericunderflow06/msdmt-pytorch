import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class MSDMT(nn.Module):
    def __init__(self,
                 timestep=10,
                 portrait_dim=32,
                 behavior_num=100 + 1,
                 behavior_emb_dim=16,
                 behavior_maxlen=64,
                 behavior_dim=32,
                 network_dim=32,
                 dropout=0.5):
        super(MSDMT, self).__init__()

        self.timestep = timestep
        self.dropout = dropout
        self.portrait_dim = portrait_dim
        self.behavior_num = behavior_num
        self.behavior_emb_dim = behavior_emb_dim
        self.behavior_maxlen = behavior_maxlen
        self.behavior_dim = behavior_dim
        self.network_dim = network_dim

        # portrait network
        self.portrait_lstm = nn.LSTM(input_size=self.portrait_dim, hidden_size=self.portrait_dim, batch_first=True)
        self.portrait_norm = nn.LayerNorm(self.portrait_dim)
        self.portrait_dense = nn.Linear(self.portrait_dim, self.portrait_dim, bias=False)

        # behavior network
        self.behavior_embedding = nn.Embedding(num_embeddings=self.behavior_num, embedding_dim=self.behavior_emb_dim, padding_idx=0)
        self.behavior_conv1d = nn.Conv1d(in_channels=self.behavior_emb_dim, out_channels=self.behavior_dim, kernel_size=3, padding=1)
        self.behavior_lstm = nn.LSTM(input_size=self.behavior_dim, hidden_size=self.behavior_dim, batch_first=True)
        self.behavior_norm = nn.LayerNorm(self.behavior_dim)
        self.behavior_dense = nn.Linear(self.behavior_dim, self.behavior_dim, bias=False)

        # graph network
        self.gcn_conv = GCNConv(in_channels=self.portrait_dim + self.behavior_dim, out_channels=self.network_dim)
        self.gcn_dropout = nn.Dropout(p=self.dropout)
        self.network_dense = nn.Linear(self.network_dim, self.network_dim)

        # output layers
        self.output1 = nn.Linear(self.network_dim, 1)
        self.output2 = nn.Linear(self.network_dim, 1)

    def forward(self, inputs):
        U, B, A = inputs  # U: user features, B: behavior sequence, A: adjacency matrix

        # portrait network
        H, _ = self.portrait_lstm(U)
        H = H[:, -1, :]  # last time step
        H = self.portrait_norm(H)
        H = F.relu(self.portrait_dense(H))

        # behavior network
        B_emb = self.behavior_embedding(B)  # shape: (batch, behavior_maxlen, behavior_emb_dim)
        B_emb = B_emb.permute(0, 2, 1)  # switch to (batch, channels, time)
        B_conv = F.relu(self.behavior_conv1d(B_emb))  # shape: (batch, behavior_dim, behavior_maxlen)
        B_pooled = torch.mean(B_conv, dim=2)  # global average pooling (batch, behavior_dim)
        B_pooled = B_pooled.unsqueeze(1).repeat(1, self.timestep, 1)  # shape: (batch, timestep, behavior_dim)
        O, _ = self.behavior_lstm(B_pooled)  # LSTM
        O = O[:, -1, :]  # last time step
        O = self.behavior_norm(O)
        O = F.relu(self.behavior_dense(O))

        # concatenate portrait and behavior features
        X = torch.cat([H, O], dim=-1)

        # graph network
        X_dense, mask = to_dense_batch(X, batch=A)  # convert sparse graph to dense batch
        V = F.relu(self.gcn_conv(X_dense, A))
        V = self.gcn_dropout(V)
        V = F.relu(self.network_dense(V))

        # outputs
        output1 = torch.sigmoid(self.output1(V))
        output2 = self.output2(V)
        return output1, output2
