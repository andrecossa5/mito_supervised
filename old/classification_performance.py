"""
Visualization of clones and samples classification performances.
"""

# Code
import sys
import os
import re
import pickle
from scipy.stats import pearsonr
from mito_utils.preprocessing import *
from mito_utils.diagnostic_plots import *
from mito_utils.heatmaps_plots import *
from mito_utils.utils import *
from mito_utils.plotting_base import *
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use('macOSX')


##

# Args
path_main = '/Users/IEO5505/Desktop/mito_bench/'
path_viz = '/Users/IEO5505/Desktop/MI_TO/images_DIPA/images' # DIPA here

# Set paths
path_data = os.path.join(path_main, 'data')
path_output = os.path.join(path_main, 'results', 'supervised_clones', 'output')
path_report = os.path.join(path_main, 'results', 'supervised_clones', 'reports')

# Create report
L = []
pickles = [ x for x in os.listdir(path_output) if bool(re.search('.pickle', x)) ]
for p in pickles:
    with open(os.path.join(path_output, p), 'rb') as f:
        d = pickle.load(f)
    L.append(d['performance_df'])

# Concat and save
pd.concat(L).to_csv(os.path.join(path_report, 'report_f1.csv'))


##


# Load report
clones = pd.read_csv(os.path.join(path_report, 'report_f1.csv'), index_col=0)


##


############## Ranking and technical summary

# Plot
metric_colors = {'precision':'#F0D290', 'recall':'#DE834D', 'f1':'#A3423C', 'AUCPR':'#5C4040'}

# Data
df_samples = (
    clones
    .assign(job=lambda x: x['filtering'] + '|' + x['dimred'] + '|' + x['model'] + '|' + x['tuning'])
    .groupby(['sample', 'job'])
    .agg('median')
    .reset_index(level=1)
)
n_pred = df_samples.shape[0]
df_samples[['filtering', 'dimred', 'model', 'tuning']] = df_samples['job'].str.split('|', expand=True)
df_samples = df_samples.drop(columns=['job']).reset_index(drop=True)
n_models = int(df_samples['model'].unique().size)

# Cahnge names
df_samples = df_samples.assign(feature_type=lambda x: x['filtering'] + '_' + x['dimred'] )
conversion_names = {
    'MQuad_no_dimred' : 'MQuad',
    'ludwig2019_no_dimred' : 'ludwig2019',
    'miller2022_no_dimred' : 'miller2022',
    'pegasus_PCA' : 'pegasus (PCA)',
    'pegasus_no_dimred' : 'pegasus'
}
df_samples['feature_type'] = df_samples['feature_type'].map(lambda x: conversion_names[x])

##

# Ranking features plot
fig, ax = plt.subplots(figsize=(8, 8))

# Feat type ranking plot
df_ = (
    df_samples.loc[:, ['f1', 'precision', 'recall', 'AUCPR', 'filtering', 'dimred', 'feature_type']]
    .drop(columns=['dimred', 'filtering'])
)
n_features = int(df_['feature_type'].unique().size)

# Order
feat_order = (
    df_.groupby('feature_type')
    .median()
    .assign(rank_col=lambda x: (x['f1'] + x['AUCPR']) / 2)
    .sort_values('rank_col', ascending=False)
    .index
)
df_ = df_.melt(id_vars='feature_type', var_name='metric')

# # Ax
sns.barplot(
    data=df_, x='value', y='feature_type', hue='metric', 
    order=feat_order,
    orient='h', 
    palette=metric_colors.values(), 
    errcolor='k',
    errwidth=.8,
    capsize=.05,
    saturation=1,
    ax=ax,
    edgecolor='k',
    linewidth=.5
)
format_ax(xlabel='Metric value', ylabel='', ax=ax, 
    title=f'Overall performance on {int(n_pred)} (aggregated, one-vs-rest) predictions \n (samples: 4, feature-types: {n_features}, models:{n_models})'
)
ax.legend(frameon=False)
ax.spines[['right', 'top', 'left']].set_visible(False)
add_legend(
    label='Metric',
    colors=metric_colors, 
    ax=ax,
    loc='lower right',
    bbox_to_anchor=(.98, .02),
    ncols=1,
    artists_size=9,
    label_size=10,
    ticks_size=9
)

# Annotate
for i, x in enumerate(feat_order):
    if x != 'pegasus (PCA)':
        v = df_samples.query("feature_type == @x")["n_features"]
        n = v.size
        median_n_vars = f'{int(np.median(v))}'
    else:
        median_n_vars = '30 (PCs)' 
    ax.text(1.05, i, f'Median (n={n}) \n n features: {median_n_vars}')


# Save
fig.tight_layout()
fig.savefig(os.path.join(path_viz, 'feature_type_ranking.png'))

##############


##


############## Top 3 models

# Data
df_ = (
    clones
    .groupby(['sample', 'comparison'])
    .apply(lambda x: 
        x
        .iloc[:,:6]
        .sort_values('AUCPR', ascending=False)
        .head(3)
    )
    .droplevel(2)
    .reset_index()
)


# Viz
samples_order = ['AML_clones', 'MDA_clones', 'MDA_lung', 'MDA_PT']

# Three good
fig = plt.figure(figsize=(14,4*2))
gs = GridSpec(2, 3, fig, width_ratios=[4, 8, 6])

for i, sample in enumerate(samples_order):
    
    # Add ax
    if sample != 'MDA_PT':
        ax = fig.add_subplot(gs[0, i])
    else:
        ax = fig.add_subplot(gs[1, :])
    
    # Data
    df_sample = df_.query('sample == @sample')
    n_clones = df_sample['comparison'].unique().size
    medians = df_sample.groupby('comparison')['AUCPR'].median()
    std = df_sample.groupby('comparison')['AUCPR'].std()
    prevalences = (
        clones.query('sample == @sample')
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
        ax.plot(j, medians[clone], 'o', c=colors[j], markersize=sizes[j])
        plt.errorbar(j, medians[clone], yerr=std[clone], c='k')
        
    ax.set(ylim=(-0.1,1.1))
    ax.hlines(0.7, 0, n_clones, 'k', 'dashed')
    format_ax(
        ax, 
        title=sample, 
        xlabel='Clones (ranked by AUCPR)',
        ylabel='AUCPR',
        xticks=[],
        xticks_size=6
    )
    ax.spines[['top', 'right']].set_visible(False)
    
    # Annotate
    df_prevalence = (
        clones.query('sample == @sample')
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
        f'AUCPR: {medians.median():.2f} +-{medians.std():.2f}', 
        transform=ax.transAxes
    )
    
# Save
fig.suptitle('Top 3 models')
fig.tight_layout()
fig.savefig(os.path.join(path_viz, 'top3_models_performances_good_samples.png'))

##############


##


############## Intrinsic MT-SNVs variation limits lentiviral clones detection

# Data
df_top3 = (
    clones
    .groupby(['sample', 'comparison'])
    .apply(lambda x: 
        x
        .iloc[:,:6]
        .sort_values('AUCPR', ascending=False)
        .head(3)
    )
    .droplevel(2)
     .sort_values(by='AUCPR', ascending=False)
     .join(
         clones.loc[:, 
             ['clone_prevalence', 'sample', 'comparison']
         ].set_index(['sample', 'comparison']), 
     )
     .reset_index()
     .drop_duplicates()
)

# Calculate correlations
corr_all = pearsonr(clones['AUCPR'], clones['clone_prevalence']) 
corr_top3 = pearsonr(df_top3['AUCPR'], df_top3['clone_prevalence']) 

##

# Plot them all
fig, ax = plt.subplots(figsize=(9,5.5))

t_high=0.1
t_low=0.01

low_p_good_clones = (
    clones
    .groupby(['sample', 'comparison'])
    .apply(lambda x: x['clone_prevalence'].unique()[0]<t_low and (x['AUCPR']>0.7).any() )
    .to_frame('to_retain').reset_index()
    .loc[lambda x: x['to_retain']]
    ['comparison'].unique()
)
high_p_bad_clones = (
    clones
    .groupby(['sample', 'comparison'])
    .apply(lambda x: x['clone_prevalence'].unique()[0]>t_high and not (x['AUCPR']>0.5).any() )
    .to_frame('to_retain').reset_index()
    .loc[lambda x: x['to_retain']]
    ['comparison'].unique()
)

assert len(high_p_bad_clones) == 0

ax.plot(
    clones.query('clone_prevalence<@t_low and AUCPR>0.7')['clone_prevalence'],
    clones.query('clone_prevalence<@t_low and AUCPR>0.7')['AUCPR'],
    'o', c='#00B206', markersize=3
)
ax.plot(
    clones.query('clone_prevalence>@t_low or AUCPR<0.7')['clone_prevalence'],
    clones.query('clone_prevalence>@t_low or AUCPR<0.7')['AUCPR'],
    'o', c='k', markersize=1
)
ax.spines[['right', 'top']].set_visible(False)

format_ax(ax, xlabel='Clonal prevalence (pr)', ylabel='AUCPR', 
    title=f'All (one-vs-rest, single-clones) predictions (n={clones.shape[0]})'
)

# Annotate
ax.text(0.35, 0.25, 
    f'Pearson\'s r: {corr_all[0]:.2f} (p={corr_all[0]:.2e})', 
    transform=ax.transAxes
)
ax.text(0.35, 0.2, 
    f'n clones pr>{t_high}, all models AUCPR<0.5: {high_p_bad_clones.size}', 
    transform=ax.transAxes, fontdict={'color':'r'}
)
ax.text(0.35, 0.15, 
    f'n clones pr<{t_low}, 1 model AUCPR>0.9: {low_p_good_clones.size}', 
    transform=ax.transAxes, fontdict={'color':'#00B206'}
)

# Format
plt.subplots_adjust(left=.25, right=.75)
fig.savefig(os.path.join(path_viz, 'good_and_bad.png'))


##