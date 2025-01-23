import os
import shutil
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
import copy

seed_value = 2021
torch.manual_seed(seed_value)
np.random.seed(seed_value)

lr = 0.0001
epochs = 500
alpha = 0.5
beta = 0.5
timestep = 10
maxlen = 64
portrait_dim = 32
behavior_num = 101
behavior_emb_dim = 16
behavior_dim = 32
network_dim = 32
dropout = 0.5

def data_process(timestep=10, maxlen=84):
    df_U = pd.read_csv('../data/sample_data_player_portrait.csv')
    df_B = pd.read_csv('../data/sample_data_behavior_sequence.csv')
    df_G = pd.read_csv('../data/sample_data_social_network.csv')
    df_Y = pd.read_csv('../data/sample_data_label.csv')
    U = df_U.drop(['uid','ds'], axis=1).values
    U = U.reshape(-1, timestep, U.shape[-1])
    B_list = df_B['seq'].apply(lambda x: x.split(',') if pd.notna(x) else []).values
    B_seq = torch.nn.utils.rnn.pad_sequence([torch.tensor(list(map(int, seq))) for seq in B_list],
                                            batch_first=True,
                                            padding_value=0)
    # Fix for reshaping
    leftover = B_seq.numel() % (timestep * maxlen)
    if leftover > 0:
        B_seq = B_seq[:-leftover]
    B_seq = B_seq.reshape(-1, timestep, 84)
    G = nx.from_pandas_edgelist(df=df_G, source='src_uid', target='dst_uid', edge_attr=['weight'])
    A = nx.adjacency_matrix(G)
    edge_index, edge_weight = from_scipy_sparse_matrix(A)
    y1 = df_Y['churn_label'].values.reshape(-1,1).astype(np.float32)
    y2 = np.log(df_Y['payment_label'].values + 1).reshape(-1,1).astype(np.float32)
    print('U:', U.shape)
    print('B:', B_seq.shape)
    print('A shape:', A.shape)
    print('y1:', y1.shape, 'y2:', y2.shape)
    return U, B_seq, (edge_index, edge_weight), y1, y2

U, B, A_data, y1, y2 = data_process(timestep=timestep, maxlen=maxlen)
edge_index, edge_weight = A_data
N = U.shape[0]

class PortraitNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PortraitNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dense = nn.Linear(hidden_dim, hidden_dim, bias=False)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n.squeeze(0)
        h = self.layernorm(h)
        h = F.relu(self.dense(h))
        return h

class BehaviorNet(nn.Module):
    def __init__(self, behavior_num, emb_dim, maxlen, timestep, behavior_dim):
        super(BehaviorNet, self).__init__()
        self.emb = nn.Embedding(num_embeddings=behavior_num, embedding_dim=emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=emb_dim, out_channels=behavior_dim, kernel_size=3, padding=1)
        self.layernorm = nn.LayerNorm(behavior_dim)
        self.dense = nn.Linear(behavior_dim, behavior_dim, bias=False)
        self.timestep = timestep
        self.behavior_dim = behavior_dim
        self.maxlen = maxlen
        self.lstm = nn.LSTM(input_size=behavior_dim, hidden_size=behavior_dim, batch_first=True)
    def forward(self, x):
        N, T, M = x.size()
        x = x.view(N*T, M)
        x = self.emb(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = F.relu(x)
        x = torch.mean(x, dim=2)
        x = x.view(N, T, self.behavior_dim)
        _, (h_n, _) = self.lstm(x)
        h = h_n.squeeze(0)
        h = self.layernorm(h)
        h = F.relu(self.dense(h))
        return h

class NetworkNet(nn.Module):
    def __init__(self, input_dim, network_dim, dropout=0.5):
        super(NetworkNet, self).__init__()
        self.gcn = GCNConv(in_channels=input_dim, out_channels=network_dim)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(network_dim, network_dim)
    def forward(self, x, edge_index, edge_weight=None):
        x = self.gcn(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.relu(self.dense(x))
        return x

class MSDMT(nn.Module):
    def __init__(self, input_dim_portrait, portrait_dim, behavior_num, behavior_emb_dim, maxlen, timestep, behavior_dim, network_dim, dropout=0.5):
        super(MSDMT, self).__init__()
        self.portrait_net = PortraitNet(input_dim=input_dim_portrait, hidden_dim=portrait_dim)
        self.behavior_net = BehaviorNet(behavior_num=behavior_num, emb_dim=behavior_emb_dim, maxlen=maxlen, timestep=timestep, behavior_dim=behavior_dim)
        self.network_net = NetworkNet(input_dim=portrait_dim+behavior_dim, network_dim=network_dim, dropout=dropout)
        self.output1 = nn.Linear(network_dim, 1)
        self.output2 = nn.Linear(network_dim, 1)
    def forward(self, U, B, edge_index, edge_weight=None):
        H = self.portrait_net(U)
        O = self.behavior_net(B)
        X = torch.cat([H, O], dim=1)
        V = self.network_net(X, edge_index, edge_weight)
        out1 = torch.sigmoid(self.output1(V))
        out2 = self.output2(V)
        return out1, out2

U_torch = torch.tensor(U, dtype=torch.float32)
B_torch = torch.tensor(B, dtype=torch.long)
y1_torch = torch.tensor(y1, dtype=torch.float32)
y2_torch = torch.tensor(y2, dtype=torch.float32)
portrait_input_dim = U_torch.shape[-1]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)
fold_idx = 1

for train_index, test_index in kfold.split(U, y1):
    print("\nFold", fold_idx)
    fold_idx += 1
    train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=seed_value)
    mask_train = np.zeros(N, dtype=bool)
    mask_val = np.zeros(N, dtype=bool)
    mask_test = np.zeros(N, dtype=bool)
    mask_train[train_index] = True
    mask_val[val_index] = True
    mask_test[test_index] = True
    model = MSDMT(
        input_dim_portrait=portrait_input_dim,
        portrait_dim=portrait_dim,
        behavior_num=behavior_num,
        behavior_emb_dim=behavior_emb_dim,
        maxlen=maxlen,
        timestep=timestep,
        behavior_dim=behavior_dim,
        network_dim=network_dim,
        dropout=dropout
    )
    optimizer = Adam(model.parameters(), lr=lr)
    patience = 5
    best_val_loss = float('inf')
    best_epoch = 0
    no_improvement = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    U_torch_dev = U_torch.to(device)
    B_torch_dev = B_torch.to(device)
    y1_torch_dev = y1_torch.to(device)
    y2_torch_dev = y2_torch.to(device)
    edge_index_dev = edge_index.to(device)
    edge_weight_dev = edge_weight.to(device) if edge_weight is not None else None
    best_state_dict = None
    for epoch_i in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out1, out2 = model(U_torch_dev, B_torch_dev, edge_index_dev, edge_weight_dev.float())
        loss_churn = F.binary_cross_entropy(out1[mask_train], y1_torch_dev[mask_train])
        loss_payment = F.mse_loss(out2[mask_train], y2_torch_dev[mask_train])
        loss = alpha*loss_churn + beta*loss_payment
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out1, val_out2 = model(U_torch_dev, B_torch_dev, edge_index_dev, edge_weight_dev.float())
            val_loss_churn = F.binary_cross_entropy(val_out1[mask_val], y1_torch_dev[mask_val])
            val_loss_payment = F.mse_loss(val_out2[mask_val], y2_torch_dev[mask_val])
            val_loss = alpha*val_loss_churn + beta*val_loss_payment
        print("Epoch", epoch_i, "Train Loss:", loss.item(), "Val Loss:", val_loss.item())
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch_i
            no_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping.")
                break
    print("Best Val Loss at epoch", best_epoch, ":", best_val_loss)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        test_out1, test_out2 = model(U_torch_dev, B_torch_dev, edge_index_dev, edge_weight_dev.float())
        test_loss_churn = F.binary_cross_entropy(test_out1[mask_test], y1_torch_dev[mask_test])
        test_loss_payment = F.mse_loss(test_out2[mask_test], y2_torch_dev[mask_test])
        test_loss = alpha*test_loss_churn + beta*test_loss_payment
    print("Fold test loss:", test_loss.item(), "(churn=", test_loss_churn.item(), "payment=", test_loss_payment.item(), ")")
print("Done.")
