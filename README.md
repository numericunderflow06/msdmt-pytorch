# msdmt-pytorch

This repo is the pytorch implementation of Multi-source Data Multi-task Learning for Profiling Players in Online Games (MSDMT) [[PDF](https://ieee-cog.org/2020/papers/paper_45.pdf)].

It is adapted from the TF2.0 implementation [[code] (https://github.com/fuxiAIlab/MSDMT)]

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
- `src/`: implementations of MSDMT.
  - `model.py`: the code for model.
  - `main.py`: the code for pipeline.

## Requirements
TODO

## Training
```
$ cd src
$ python main.py 
```

