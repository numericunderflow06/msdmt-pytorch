Some ideas behind the conversion from TensorFlow to pytorch:

- We replace ``Sequential`` of Tensorflow by ``nn.Module`` in pytorch, where the layers are coordinated in ``forward``
- We replace Spektral's ``GCNConv`` with ``torch_geometric``'s ``GCNConv`` for graphs
- We replace TensorFlow's ``GlobalAveragePooling1D`` by ``torch.mean`` over the time dimension

Some other logics have been updated (to document)
