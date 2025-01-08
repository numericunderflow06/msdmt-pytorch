# msdmt-pytorch

This repo is the pytorch implementation of Multi-source Data Multi-task Learning for Profiling Players in Online Games (MSDMT) [[PDF](https://ieee-cog.org/2020/papers/paper_45.pdf)].

It is adapted from the TF2.0 implementation (https://github.com/fuxiAIlab/MSDMT)

The description of the authors:
MSDMT is a novel Multi-source Data Multi-task Learning approach for profiling players with both player churn and payment prediction in online games. 
On the one hand, MSDMT considers that heterogeneous multi-source data, including player portrait tabular data, behavior sequence sequential data, and social network graph data, can complement each other for a better understanding of each player.
On the other hand, MSDMT considers the significant correlation between the player churn and payment that can interact and complement each other.

## Folders
- `data/`: data of MSDMT (**randomly generated sample data** to show the data format, **not the real data**).
  - `sample_data_player_portrait.csv`: the sample data for player portrait.
  - `sample_data_behavior_sequence.csv`: the sample data for behavior sequence.
  - `sample_data_social_network.csv`: the sample data for social network.
  - `sample_data_label.csv`: the sample data for label, where label1 is churn label (binary classification) and label2 is payment label (regression).
- `src/`: implementations of MSDMT in **pytorch**.
  - `model.py`: the code for model.
  - `main.py`: the code for pipeline.
- `src-tf/`: implementations of MSDMT in TensorFlow 2.0.
  - `model.py`: the code for model.
  - `main.py`: the code for pipeline.
## Requirements
Core libraries: 
- numpy 
- pandas
- scikit-learn
- networkx
- torch
  
Other libraries:
- torch-scatter
- torch-sparse
- torch-cluster
- torch-spline-conv
- torch-geometric

## Training
```
$ cd src
$ python main.py 
```

## Instructions to compare with TensorFlow

### Requirements
The authors indicated the following dependencies:
The code has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):
- tensorflow == 2.1.0
- spektral ==1.0.3
- numpy == 1.18.2
- pandas == 0.23.4
- sklearn == 0.19.1
  
However I have not tested it with their settings, and it might be deprecated
### Training
```
$ cd src-tf
$ python main.py
```

