"""
Visualization of individual classification performances.
"""

# Code
import os
import re
import pickle
import pandas as pd
import numpy as np
from plotting_utils._utils import *
from plotting_utils._plotting_base import *
from matplotlib.gridspec import GridSpec
matplotlib.use('macOSX')


##

# Args
path_supervised = '/Users/IEO5505/Desktop/mito_bench/results/supervised_clones'

# Create report of classification performance
# L = []
# pickles = [ 
#     x for x in os.listdir(os.path.join(path_supervised, 'output')) \
#     if bool(re.search('.pickle', x)) 
# ]
# for p in pickles:
#     try:
#         with open(os.path.join(path_supervised, 'output', p), 'rb') as f:
#             d = pickle.load(f)
#     except:
#         print('Logit, still only stub')
#     L.append(d['performance_df'])
# 
# # Concat and save
# pd.concat(L).to_csv(os.path.join(path_supervised, 'report_f1.csv'))


##


# Save MT-SNVs subsets
# d = {}
# pickles = [ 
#     x for x in os.listdir(os.path.join(path_supervised, 'output')) \
#     if bool(re.search('.pickle', x)) 
# ]
# for p in pickles:
#     try:
#         with open(os.path.join(path_supervised, 'output', p), 'rb') as f:
#             r = pickle.load(f)
#     except:
#         print('stub file')
#     k = r['performance_df'][['sample', 'filtering']].drop_duplicates().values[0]
#     k = tuple(k)
#     if k not in d:
#         _ = list(r['trained_models'].keys())[0]
#         d[k] = r['trained_models'][_]['variants'].to_list()
# 
# # Save
# with open(os.path.join(path_supervised, 'variants.pickle'), 'wb') as f:
#     pickle.dump(d, f)


##


# Read report
df = pd.read_csv(os.path.join(path_supervised, 'report_f1.csv'), index_col=0)

# Precision analysis
# Corr clone_prevalence and performance
fig, ax = plt.subplots(figsize=(3.5,4))

df['clone_prevalence_bins'] = pd.cut(np.round(df['clone_prevalence'], 2), bins=4)
bins = df['clone_prevalence_bins'].cat.categories.to_list()
bins[0] = "(0,0.18]"
box(df, 'clone_prevalence_bins', 'precision', ax=ax, c='white')
strip(df, 'clone_prevalence_bins', 'precision', ax=ax, c='k', s=2)
corr = np.corrcoef(df['clone_prevalence'], df['precision'])
format_ax(
    ax, reduce_spines=True, rotx=45, xlabel='Clone prevalence bin', ylabel='Precision',
    xticks=bins, title=f'Pearson\'s r: {corr[0,1]:.2f}'
)
fig.tight_layout()
fig.savefig(os.path.join(
    path_supervised, 'precision_vs_clone_prevalence.png'),
    dpi=300
)

##

# Corr clone_prevalence and performance
fig, ax = plt.subplots(figsize=(3.5,4))

box(df, 'n_clones_analyzed', 'precision', ax=ax, c='white')
strip(df, 'n_clones_analyzed', 'precision', ax=ax, c='k', s=2)
corr = np.corrcoef(df['n_clones_analyzed'], df['precision'])
format_ax(
    ax, reduce_spines=True, xlabel='n clones analyzed', ylabel='Precision', 
    title=f'Pearson\'s r: {corr[0,1]:.2f}'
)
fig.tight_layout()
fig.savefig(os.path.join(
    path_supervised, 'precision_vs_n_clones.png'),
    dpi=300
)


##


# Individual metrics
metric = 'AUPRC'

# All models
s = df.groupby(['sample', 'comparison'])[metric].median()
np.sum(s>=.7)/s.size

# Top 3
top_3 = (
    df.groupby(['sample', 'comparison'])
    .apply(lambda x: x.sort_values(metric, ascending=False).head(3))
    [['n_clones_analyzed', 'clone_prevalence', 'ncells_clone', 'ncells_sample',
      'filtering', 'model', 'AUPRC', 'precision', 'recall', 'f1']]
    .droplevel(2).reset_index()
)


##


# Combos
# combo = ['sample', 'model', 'filtering']
# combo = ['model', 'filtering']
combo = ['filtering']
# combo = ['model']

# Viz all_models, grouped by some combos
df_ = (
    df[['precision', 'recall', 'f1', 'AUPRC', 'model', 'filtering', 'sample']]
    .groupby(combo)[['precision', 'recall', 'f1', 'AUPRC']]
    .median().reset_index()
    .sort_values(metric, ascending=False)
)
df_['cat'] = df_[combo].apply(lambda x: '_'.join(x), axis=1)
df_ = df_.assign(
    rank_precision=lambda x: x['precision'].rank(ascending=False),
    rank_recall=lambda x: x['recall'].rank(ascending=False),
    rank_f1=lambda x: x['f1'].rank(ascending=False),
    rank_AUPRC=lambda x: x['AUPRC'].rank(ascending=False)
)
# Top combo
df_['final_rank'] = df_[df_.columns[df_.columns.str.contains('rank')]].mean(axis=1)
df_.sort_values('final_rank', ascending=True)


##


# Viz aggregate combos
fig, ax = plt.subplots(figsize=(5,7))
x = metric
ax.hlines(y=df_.index, xmin=0, xmax=df_[x], color='k')
ax.plot(df_[x], df_[x].index, "o", color='darkred', markeredgecolor='k', markersize=5)
ax.invert_yaxis()
format_ax(
    ax, xlabel='value', yticks=df_['cat'], 
    title=f'{"_".join(combo)}\n median AUPRC: {df_["AUCPR"].median():.2f}',
    yticks_size=6
)
ax.spines[['right', 'top', 'left']].set_visible(False)
fig.tight_layout()
plt.show()


##


# Viz top models
samples_order = ['AML_clones', 'MDA_clones', 'MDA_lung', 'MDA_PT']
fig = plt.figure(figsize=(13,7))
gs = GridSpec(2, 3, fig, width_ratios=[4, 8, 11])

for i, sample in enumerate(samples_order):
    
    # Add ax
    if sample != 'MDA_PT':
        ax = fig.add_subplot(gs[0,i])
    else:
        ax = fig.add_subplot(gs[1,:])
    
    # Data
    df_sample = top_3.query('sample == @sample')
    n_clones = df_sample['comparison'].unique().size
    medians = df_sample.groupby('comparison')[metric].median()
    std = df_sample.groupby('comparison')[metric].std()
    prevalences = (
        top_3.query('sample == @sample')
        .loc[:, ['comparison', 'clone_prevalence']]
        .drop_duplicates()
        .set_index('comparison')['clone_prevalence']
    )
    clone_order = medians.sort_values(ascending=False).index
    colors = sns.color_palette('inferno_r', n_colors=n_clones)
    sizes = np.interp(
        prevalences[clone_order],
        (prevalences[clone_order].min(), prevalences[clone_order].max()), 
        (5, 20)
    )

    # Plot
    for j, clone in enumerate(clone_order):
        ax.plot(j, medians[clone], 'o', c=colors[j], markersize=sizes[j], markeredgecolor='k')
        plt.errorbar(j, medians[clone], yerr=std[clone], c='k')
        
    ax.set(ylim=(-0.1,1.1))
    ax.axhline(0.7, 0, n_clones, color='k')
    format_ax(
        ax, 
        title=sample, 
        xlabel=f'Clones (ranked by {metric})',
        ylabel=metric,
        xticks=[],
        xticks_size=6
    )
    ax.spines[['top', 'right']].set_visible(False)
    
    # Annotate
    df_prevalence = (
        top_3.query('sample == @sample')
        .loc[:, ['ncells_sample', 'clone_prevalence', 'comparison']]
        .drop_duplicates()
    )
    n_cells = df_prevalence['ncells_sample'].unique()[0]
    SH = -np.sum(np.log10(df_prevalence['clone_prevalence']) * df_prevalence['clone_prevalence'])
    
    ax.text(.05, .71, '0.7', transform=ax.transAxes)
    ax.text(.05, .23, f'n_cells: {int(n_cells)}', 
        transform=ax.transAxes
    )
    ax.text(.05, .17, f'n_clones: {int(n_clones)}', 
        transform=ax.transAxes
    )
    ax.text(.05, .11, f'Shannon Entropy (SH): {SH:.2f}', 
        transform=ax.transAxes
    )
    ax.text(.05, .05, 
        f'{metric}: {medians.median():.2f} +-{medians.std():.2f}', 
        transform=ax.transAxes
    )
    
# Save
fig.suptitle(f'{metric} top 3 models')
fig.tight_layout()
fig.savefig(os.path.join(
    path_supervised, f'{metric}_top3_models_performances_good_samples.png'),
    dpi=300
)


##
