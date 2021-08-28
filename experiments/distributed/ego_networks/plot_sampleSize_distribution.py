import os
import pickle
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=2)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd



def read_sampleSizes(inpath, datalist, alphalist):
    names = []
    list_sampleSizeDistribution = []
    for alpha in alphalist:
        for data in datalist:
            list_sampleSize = pickle.load(open(os.path.join(inpath, f'sample_num_distribution_{data}_alpha{alpha}.pkl'), 'rb'))
            df = pd.DataFrame()
            df['dataset'] = data
            df['alpha'] = alpha
            df['sample size'] = list_sampleSize
            df['client'] = list(range(1, len(list_sampleSize)+1))
            names.append(f'{data} (alpha={alpha})')
            list_sampleSizeDistribution.append(df)
    return list_sampleSizeDistribution, names


def grid_plot_distribution(list_sampleSizeDistribution, names, plotfile):
    ncols = 4
    nrows = 2
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20, 10))

    for i in range(len(list_sampleSizeDistribution)):
        ax = plt.subplot(gs[i])
        df = list_sampleSizeDistribution[i]
        sns.barplot(x='client', y='sample size', data=df)
        plt.title(names[i], fontsize=20)

    plt.tight_layout()
    plt.savefig(plotfile)


def test(inpath, filename1, filename2):
    # df = pd.DataFrame({'sample size': [561, 1, 20, 2, 187, 12, 3, 3, 10, 1, 105, 93, 88, 66, 73, 60, 57, 105, 80, 73], 
    #     'alpha': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]})

    df1 = pd.DataFrame()
    df1['client'] = list(range(1, 11))
    df1['alpha'] = 0.1
    df1['sample size'] = pickle.load(open(os.path.join(inpath, filename1), 'rb'))
    print(df1)
    df2 = pd.DataFrame()
    df2['client'] = list(range(1, 11))
    df2['alpha'] = 10.0
    df2['sample size'] = pickle.load(open(os.path.join(inpath, filename2), 'rb'))
    print(df2)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='client', y='sample size', data=df1, color="royalblue")
    plt.tight_layout()
    plt.savefig("../plot_sampleSizeDistribution_alpha0.1.pdf")


    plt.figure(figsize=(8, 5))
    sns.barplot(x='client', y='sample size', data=df2, color="royalblue")
    plt.tight_layout()
    plt.savefig("../plot_sampleSizeDistribution_alpha10.pdf")


if __name__ == '__main__':
    inpath = './visualization'
    # datalist = ['cora', 'citeseer', 'PubMed', 'DBLP']
    # alphalist = [0.1, 10.0]
    # list_sampleSizeDistribution, names = read_sampleSizes(inpath, datalist, alphalist)
    # plotfile = './visualization/plot_sampleSizeDistribution.pdf'
    # grid_plot_distribution(list_sampleSizeDistribution, names, plotfile)

    test(inpath, 'sample_num_distribution_cora_alpha0.1.pkl', 'sample_num_distribution_cora_alpha10.0.pkl')