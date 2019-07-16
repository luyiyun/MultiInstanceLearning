import os

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch


def tidy_train(train_file):
    train_df = pd.read_csv(train_file)
    mul_index = pd.MultiIndex.from_tuples(
        [c.split('_') for c in train_df.columns],
        names=['metric', 'phase']
    )
    train_df.columns = mul_index
    train_df = train_df.stack(level=[0, 1]).reset_index()
    train_df = train_df.rename({'level_0': 'epoch', 0: 'value'}, axis=1)
    return train_df


def plot_train(tidy_train_df):
    metrics = set(tidy_train_df['metric'].values)
    num_metrics = len(metrics)
    phases = set(tidy_train_df['phase'].values)
    num_phase = len(phases)

    _, axes = plt.subplots(nrows=num_metrics, ncols=num_phase)

    for i, m in enumerate(metrics):
        for j, p in enumerate(phases):
            ax = axes[i, j]
            subdf = tidy_train_df.loc[
                (tidy_train_df['metric'] == m) &
                (tidy_train_df['phase'] == p)
            ]
            x = subdf['epoch'].values
            y = subdf['value'].values
            ax.plot(x, y)
            ax.set_xlabel('epoch')
            ax.set_ylabel(m)
            if i == 0:
                ax.set_title(p)

    plt.show()


def main():
    # train_file = './RESULTS/train1/train.csv'
    # tidy_train_df = tidy_train(train_file)
    # plot_train(tidy_train_df)

    weighter_file = './RESULTS/train1/weigher.pth'
    weighter = torch.load(weighter_file)
    p_arr = weighter.p_tensor.cpu().numpy()
    im = plt.imshow(p_arr, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.show()



if __name__ == '__main__':
    main()
