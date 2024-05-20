"""
Multiclass and distance metric evaluation
"""

import os
import re
import pickle
from mito_utils.utils import *
from plotting_utils._plotting_base import *
matplotlib.use('macOSX')


##


# Set paths
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data') 
path_results = os.path.join(path_main, 'results', 'supervised_clones')


##


# Groupby
groupby = ['distance_metric']


# Load AUPRC results
L = []
for sample in os.listdir(os.path.join(path_results, 'distances')):
    path_ = os.path.join(path_results, 'distances', sample)
    for x in os.listdir(path_):
        if bool(re.search('metrics_kNN', x)):
            key = x.split('_')[3:]
            sample = '_'.join([key[0], key[1]])
            filtering = key[2].split('.')[0]
            with open(os.path.join(path_, x), 'rb') as f:
                d = pickle.load(f)
        L.append(
            pd.DataFrame(d).agg([np.median])
            .T.reset_index()
            .rename(columns={'index':'distance_metric'})
            .assign(sample=sample, filtering=filtering, metric='AUPRC')
        )

# Summary per distance metric
df_auprc = (
    pd.concat(L)
    .groupby('distance_metric').median()
    .rename(columns={'median':'AUPRC_median'})
    .assign(
        AUPRC_rank=lambda x: x['AUPRC_median'].rank(ascending=False)
    )
)


##


# Load kNN metrics
L = []
for sample in os.listdir(os.path.join(path_results, 'distances')):
    path_ = os.path.join(path_results, 'distances', sample)
    for x in os.listdir(path_):
        if bool(re.search('metrics_kNN', x)):
            key = x.split('_')[3:]
            sample = '_'.join([key[0], key[1]])
            filtering = key[2].split('.')[0]
            with open(os.path.join(path_, x), 'rb') as f:
                d = pickle.load(f)
        L.append(
            pd.DataFrame(d)
            .T.reset_index()
            .rename(columns={'index':'distance_metric'})
            .assign(sample=sample, filtering=filtering)
        )

# Build df
df_kNN = (
    pd.concat(L)
    .groupby(['distance_metric']).median()
    .sort_values('median_NN_purity', ascending=False)
    .assign(
        rank_kBET=lambda x: x['kBET_rejection_rate'].rank(ascending=False),
        rank_entropy=lambda x: x['median_NN_entropy'].rank(ascending=True),
        rank_purity=lambda x: x['median_NN_purity'].rank(ascending=False),
    )
)
# Final df aggregated metrics
df = df_auprc.join(df_kNN)

# Top metric
df['final_rank'] = df[df.columns[df.columns.str.contains('rank')]].mean(axis=1)
df = df.sort_values('final_rank', ascending=True).reset_index()


##


# Viz
fig, ax = plt.subplots(figsize=(4,4))
scatter(df, 'AUPRC_median', 'kBET_rejection_rate', ax=ax, s=30, marker='x')
format_ax(ax, title='Distance metrics', xlabel='kNN purity', ylabel='AUPRC')
x = df['AUPRC_median']
y = df['kBET_rejection_rate']
ta.allocate_text(fig, ax, x, y, df['distance_metric'].values, x_scatter=x, y_scatter=y,
    linecolor='black', textsize=8, max_distance=0.5, linewidth=0.5, nbr_candidates=100)
fig.tight_layout()
fig.savefig(os.path.join(path_results, 'multiclass_distances.pdf'), dpi=300)


##