import os
import shutil

import networkx as nx
import numpy as np
import pandas as pdx
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data, DataLoader
from model import MSDMT  # import the model from src/model.py

##############################
seed_value = 2021
lr = 0.0001
epochs = 500
alpha = 0.5
beta = 0.5
timestep = 10
maxlen = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##############################

# set random seed
torch.manual_seed(seed_value)
np.random.seed(seed_value)

def data_process(timestep=10, maxlen=64):
    df_U = pd.read_csv('../data/sample_data_player_portrait.csv')
    df_B = pd.read_csv('../data/sample_data_behavior_sequence.csv')
    df_G = pd.read_csv('../data/sample_data_social_network.csv')
    df_Y = pd.read_csv('../data/sample_data_label.csv')

    # user features
    U = df_U.drop(['uid', 'ds'], axis=1).values
    U = U.reshape(-1, timestep, U.shape[-1])
    U = torch.tensor(U, dtype=torch.float32)

    # behavior sequences
    B = df_B['seq'].apply(lambda x: x.split(',') if pd.notna(x) else []).values
    B = torch.tensor(
        nn.utils.rnn.pad_sequence(
            [torch.tensor(list(map(int, seq)), dtype=torch.long) for seq in B],
            batch_first=True,
            padding_value=0
        ),
        dtype=torch.long
    ).reshape(-1, timestep, maxlen)

    # social network graph
    G = nx.from_pandas_edgelist(df=df_G, source='src_uid', target='dst_uid', edge_attr=['weight'])
    A = nx.adjacency_matrix(G)
    edge_index, edge_weight = from_scipy_sparse_matrix(A)

    # labels
    y1 = torch.tensor(df_Y['churn_label'].values, dtype=torch.float32).unsqueeze(-1)
    y2 = torch.tensor(np.log(df_Y['payment_label'].values + 1), dtype=torch.float32).unsqueeze(-1)

    print('U:', U.shape)
    print('B:', B.shape)
    print('G:', A.shape)
    print('y1:', y1.shape, 'y2:', y2.shape)

    return U, B, edge_index, edge_weight, y1, y2


U, B, edge_index, edge_weight, y1, y2 = data_process(timestep=timestep, maxlen=maxlen)

# dataset preparation
N = U.shape[0]

dataset = Data(x=torch.cat((U, B), dim=-1), edge_index=edge_index, edge_attr=edge_weight, y1=y1, y2=y2)
data_loader = DataLoader([dataset], batch_size=N)

# model and training
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)

for train_index, test_index in kfold.split(U.numpy(), y1.numpy().ravel()):

    train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=seed_value)

    mask_train = torch.zeros(N, dtype=torch.bool)
    mask_val = torch.zeros(N, dtype=torch.bool)
    mask_test = torch.zeros(N, dtype=torch.bool)
    mask_train[train_index] = True
    mask_val[val_index] = True
    mask_test[test_index] = True

    model = MSDMT(timestep=timestep, behavior_maxlen=maxlen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()

    best_loss = float('inf')
    patience = 5
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output1, output2 = model([U.to(device), B.to(device), edge_index.to(device)])
        loss1 = criterion1(output1[mask_train], y1[mask_train].to(device))
        loss2 = criterion2(output2[mask_train], y2[mask_train].to(device))
        loss = alpha * loss1 + beta * loss2
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_output1, val_output2 = model([U.to(device), B.to(device), edge_index.to(device)])
            val_loss1 = criterion1(val_output1[mask_val], y1[mask_val].to(device))
            val_loss2 = criterion2(val_output2[mask_val], y2[mask_val].to(device))
            val_loss = alpha * val_loss1 + beta * val_loss2

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

        # early stopping
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            wait = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    # load the best model for evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    with torch.no_grad():
        test_output1, test_output2 = model([U.to(device), B.to(device), edge_index.to(device)])
        test_loss1 = criterion1(test_output1[mask_test], y1[mask_test].to(device))
        test_loss2 = criterion2(test_output2[mask_test], y2[mask_test].to(device))
        print(f"Test Loss: {alpha * test_loss1 + beta * test_loss2}")
