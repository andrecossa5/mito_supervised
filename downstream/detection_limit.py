"""
Further hints over the detection limit of lentiviral clones. MDA_PT only.
1. What is the relationship among the exclusivity of a clone MT-NSVs, and its AUCPR? 
2. What is the relationship among clones (diss-)imilarity (evaluated according some metric in the MT-SNVs space), and their AUCPR? 
"""

# Code
import os
import re
import pickle

from scipy.stats import pearsonr
from mito_utils.preprocessing import *
from mito_utils.clustering import *
from mito_utils.heatmaps_plots import *
from mito_utils.utils import *
from mito_utils.distances import *
from mito_utils.plotting_base import *
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use('macOSX')


##

# Set paths
path_main = '/Users/IEO5505/Desktop/mito_bench/'
path_data = os.path.join(path_main, 'data')
path_output = os.path.join(path_main, 'results', 'supervised_clones', 'output')
path_report = os.path.join(path_main, 'results', 'supervised_clones', 'reports')
path_viz = os.path.join(path_main, 
                        'results', 'supervised_clones', 'visualization', 'detection_limit_PT')
path_tmp = os.path.join(path_main, 'results', 'supervised_clones', 'downstream_files')


##


# Data
afm = read_one_sample(path_data, 'MDA_PT', with_GBC=True)
df_performance = (
    pd.read_csv(os.path.join(path_report, 'report_f1.csv'), index_col=0)
    .query('sample == "MDA_PT"')
)
clones = df_performance['comparison'].map(lambda x: x.split('_')[0]).unique()

# Best combo, median on all clones
conversion_names = {
    'MQuad_no_dimred' : 'MQuad',
    'ludwig2019_no_dimred' : 'ludwig2019',
    'miller2022_no_dimred' : 'miller2022',
    'pegasus_PCA' : 'pegasus (PCA)',
    'pegasus_no_dimred' : 'pegasus'
}
df_performance = df_performance.assign(feature_type=lambda x: x['filtering'] + '_' + x['dimred'])
df_performance['feature_type'] = df_performance['feature_type'].map(lambda x: conversion_names[x])
df_performance = df_performance.assign(combo=lambda x: x['feature_type'] + '_' + x['model'])
df_combos_on_average = (
    df_performance
    .groupby('combo')
    .agg('median')
    .sort_values('AUCPR', ascending=False)
)
top_5_combos_on_average = df_combos_on_average.index[:5].map(lambda x: x.split('_')).to_list()


##


############################## 1. What is the relationship among the exclusivity of a clone MT-NSVs, and its AUCPR? 

# Load variants
with open(os.path.join(path_tmp, 'pg_MQuad_miller_vars.pickle'), 'rb') as f:
    variants = pickle.load(f)

# Get a_cells from full afm
a_cells, _ = filter_cells_and_vars(
    afm, sample='MDA_PT', min_cell_number=10, min_cov_treshold=50, variants=variants['pegasus']
)

# Here we go
filtering = 'MQuad' # Choose one filtering method

# Compute stats: n of stably expressed MT-SNVs (min_clone_perc>=.7)
var_clones = {
    clone : 
    rank_clone_variants(
        a_cells[:, variants[filtering]], clone, min_clone_perc=.7, max_perc_rest=1
    ).index.to_list() \
    for clone in clones
}
n_stably_expressed = [ len(var_clones[clone]) for clone in clones ]

# Compute stats: n of exclusive MT-SNVs (min_clone_perc>=.7) 
var_clones = {
    clone : 
    rank_clone_variants(
        a_cells[:, variants[filtering]], clone, min_clone_perc=.7, max_perc_rest=.1
    ).index.to_list() \
    for clone in clones
}
n_exclusive = [ len(var_clones[clone]) for clone in clones ]

# Compute stats: median Jaccard Index between a clone stably expressed MT-SNVs and all the other clones

# JI matrix, all clones vs the others
var_clones = {
    clone : 
    rank_clone_variants(
        a_cells[:, variants[filtering]], clone, min_clone_perc=.7, max_perc_rest=1
    ).index.to_list() \
    for clone in clones
}

ji = lambda x,y: len(set(x) & set(y)) / len(set(x) | set(y))

n = len(clones)
J = np.ones((n,n))
for i, x in enumerate(var_clones):
    for j, y in enumerate(var_clones):
        if len(var_clones[x]) == 0 and len(var_clones[y]) == 0:
            J[i,j] = np.nan
        else:
            J[i,j] = ji(var_clones[x], var_clones[y])

# Median JI with other clones 
median_JI = np.nanmean(J, axis=0)
        
# Df and visualization
df_ = pd.DataFrame({
    'n_stably_expressed' : n_stably_expressed,
    'n_exclusive' : n_exclusive,
    'median_JI' : median_JI,
    'comparison' : [ f'{clone}_vs_rest' for clone in clones ]
})
df_ = pd.merge(df_performance, df_, on='comparison')

# Median models per clone
df_top = (
    df_
    .query('feature_type == @filtering')
    .groupby('comparison')
    .agg('median')
    .loc[:, ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'AUCPR', 
        'ncells_clone', 'clone_prevalence', 'n_features', 'n_stably_expressed', 'n_exclusive', 'median_JI'
    ]]
    .sort_values('AUCPR', ascending=False)
)

# Viz
fig, axs = plt.subplots(2,2,figsize=(8,8))

axs[0,0].plot(df_top['n_stably_expressed'], df_top['AUCPR'], 'ko')
sns.regplot(x='n_stably_expressed', y='AUCPR', data=df_top, ax=axs[0,0])
r, p = pearsonr(df_top['n_stably_expressed'], df_top['AUCPR'])
axs[0,0].set(title=f'Pearson\'r: {r:.2f} (pvalue={p:.2e})')

axs[0,1].plot(df_top['n_exclusive'], df_top['AUCPR'], 'ro')
sns.regplot(x='n_exclusive', y='AUCPR', data=df_top, ax=axs[0,1])
r, p = pearsonr(df_top['n_exclusive'], df_top['AUCPR'])
axs[0,1].set(title=f'Pearson\'r: {r:.2f} (pvalue={p:.2e})')

axs[1,0].plot(df_top['median_JI'], df_top['AUCPR'], 'bo')
sns.regplot(x='median_JI', y='AUCPR', data=df_top, ax=axs[1,0])
r, p = pearsonr(df_top['median_JI'], df_top['AUCPR'])
axs[1,0].set(title=f'Pearson\'r: {r:.2f} (pvalue={p:.2e})')

axs[1,1].plot(df_top['clone_prevalence'], df_top['AUCPR'], 'go')
sns.regplot(x='clone_prevalence', y='AUCPR', data=df_top, ax=axs[1,1])
r, p = pearsonr(df_top['clone_prevalence'], df_top['AUCPR'])
axs[1,1].set(title=f'Pearson\'r: {r:.2f} (pvalue={p:.2e})')

fig.tight_layout()
fig.savefig(os.path.join(path_viz, f'variants_picture_{filtering}.png'), dpi=500)


##


############################## 2. What is the relationship among clones 
# dis-similarities and their AUCPR? 

filtering = 'MQuad' # Choose one filtering method

# Aggregate AFM at the clone level
a = a_cells[:, variants[filtering]].copy()
clone_agg = (
    pd.DataFrame(a.X, index=a.obs_names, columns=variants[filtering])
    .assign(GBC=a.obs['GBC'].astype('str'))
    .reset_index(drop=True)
    .groupby('GBC')
    .agg('median') # Use median here
)

# Viz aggregated
fig, ax = plt.subplots(figsize=(6,6))
plot_heatmap(
    clone_agg, ax=ax, vmin=0, vmax=0.02, x_names_size=5, 
    y_names_size=5, cb=False, rank_diagonal=True
)
fig.tight_layout()
fig.savefig(
    os.path.join(path_viz, 'MDA_PT_MQuad_aggregated_clone_vars_heatmap.png'),
    dpi=500
)

# Compute median clones dissimilarities (median one-vs-rest pairwise distances)
df_dists = pd.DataFrame({
    'correlation' : np.median(pair_d(clone_agg, metric='correlation'), axis=0),
    'sqeuclidean' : np.median(pair_d(clone_agg, metric='sqeuclidean'), axis=0),
    'cosine' : np.median(pair_d(clone_agg, metric='cosine'), axis=0),
    'euclidean' : np.median(pair_d(clone_agg, metric='euclidean'), axis=0),
    'cosine' : np.median(pair_d(clone_agg, metric='cosine'), axis=0)
    }, index=clone_agg.index
)
df_dists.index = df_dists.index.map(lambda x: f'{x}_vs_rest')
df_dists = df_dists.reset_index().rename(columns={'GBC':'comparison'})

# Merge
df_ = pd.merge(df_performance, df_dists, on='comparison')

# Median models per clone
df_top = (
    df_
    .query('feature_type == @filtering')
    .groupby('comparison')
    .agg('median')
    .loc[:, ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'AUCPR', 
        'ncells_clone', 'clone_prevalence', 'correlation', 'euclidean', 'sqeuclidean', 'cosine'
    ]]
    .sort_values('AUCPR', ascending=False)
)

# Viz
fig, axs = plt.subplots(2,2,figsize=(8,8))

axs[0,0].plot(df_top['euclidean'], df_top['AUCPR'], 'ko')
sns.regplot(x='euclidean', y='AUCPR', data=df_top, ax=axs[0,0])
r, p = pearsonr(df_top['euclidean'], df_top['AUCPR'])
axs[0,0].set(title=f'Pearson\'r: {r:.2f} (pvalue={p:.2e})')

axs[0,1].plot(df_top['sqeuclidean'], df_top['AUCPR'], 'ro')
sns.regplot(x='sqeuclidean', y='AUCPR', data=df_top, ax=axs[0,1])
r, p = pearsonr(df_top['sqeuclidean'], df_top['AUCPR'])
axs[0,1].set(title=f'Pearson\'r: {r:.2f} (pvalue={p:.2e})')

axs[1,0].plot(df_top['correlation'], df_top['AUCPR'], 'bo')
sns.regplot(x='correlation', y='AUCPR', data=df_top, ax=axs[1,0])
r, p = pearsonr(df_top['correlation'], df_top['AUCPR'])
axs[1,0].set(title=f'Pearson\'r: {r:.2f} (pvalue={p:.2e})')

axs[1,1].plot(df_top['cosine'], df_top['AUCPR'], 'go')
sns.regplot(x='cosine', y='AUCPR', data=df_top, ax=axs[1,1])
r, p = pearsonr(df_top['cosine'], df_top['AUCPR'])
axs[1,1].set(title=f'Pearson\'r: {r:.2f} (pvalue={p:.2e})')

fig.tight_layout()
fig.savefig(os.path.join(path_viz, f'dists_picture_{filtering}.png'), dpi=500)


##
