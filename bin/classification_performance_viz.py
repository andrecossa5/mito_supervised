"""
Visualization of clones and samples classification performances.
"""

# Code
import os
from pysam import VariantFile
from scipy.interpolate import interp1d 
from mito_utils.preprocessing import *
from mito_utils.diagnostic_plots import *
from mito_utils.heatmaps_plots import *
from mito_utils.utils import *
from mito_utils.plotting_base import *
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use('macOSX')


##

params = {   
    'showcaps' : True,
    'fliersize': 0,
    'boxprops' : {'edgecolor': 'black', 'linewidth': .8}, 
    'medianprops': {"color": "black", "linewidth": 1.5},
    'whiskerprops':{"color": "black", "linewidth": 1.2}
}

##


# Set paths
path_data = '/Users/IEO5505/Desktop/example_mito/data/'
path_main = '/Users/IEO5505/Desktop/example_mito/results/supervised_clones/' 

# Read report
clones = pd.read_csv(
    os.path.join(path_main, 'reports', 'report_f1.csv'),
    index_col=0
)


##


############## Extended summary, aggregate by sample

# Full summary
# clones.describe().to_excel(
#     os.path.join(path_main, 'reports', 'full_aggregate_f1.xlsx')
# )

# Agg by sample and run summary
# (
#     clones
#     .assign(job=lambda x: x['filtering'] + '_' + x['dimred'] + '_' + x['model'] + '_' + x['tuning'])
#     .groupby(['sample', 'job'])
#     .agg('mean')
#     .reset_index(level=1)
#     .groupby('sample')
#     .apply(lambda x: x.sort_values('f1', ascending=False))
# ).to_excel(
#     os.path.join(path_main, 'reports', 'sample_job_aggregate_f1.xlsx')
# )

# Plot
fig, axs = plt.subplots(1,2, figsize=(13.5, 5), constrained_layout=True)

df_ = (
    clones
    .assign(job=lambda x: x['filtering'] + '_' + x['dimred'] + '_' + x['model'] + '_' + x['tuning'])
    .groupby(['sample', 'job'])
    .agg('mean')
)

# Top
box(df_.iloc[:,:5].melt(), 'variable', 'value', c='#E9E7E7', order=None, ax=axs[0], kwargs=params)
strip(df_.query('f1>=0.7').iloc[:,:5].melt(), 'variable', 'value', c='r', s=3.5, ax=axs[0])
strip(df_.query('f1<=0.7').iloc[:,:5].melt(), 'variable', 'value', c='k', s=2.5, ax=axs[0])

n_good = df_.query('f1>=0.7').shape[0]
format_ax(axs[0], ylabel='value', xticks_size=10, rotx=90)
add_legend(
    label='Model',
    colors={'f1 >= 0.7':'r', 'f1 < 0.7':'k'}, 
    ax=axs[0],
    loc='upper left',
    bbox_to_anchor=(1,1),
    ncols=1,
    label_size=12,
    ticks_size=10
)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_visible(False)

# Bottom
df_['feature_type'] = df_.reset_index(level=1)['job'].map(lambda x: '_'.join(x.split('_')[:-2])).values
feat_type_colors = create_palette(df_, 'feature_type', 'Spectral')

box(df_.iloc[:,:5].melt(), 'variable', 'value', c='#E9E7E7', order=None, ax=axs[1], kwargs=params)
strip(
    df_.loc[:,['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'feature_type']]
    .melt(id_vars='feature_type'),
    'variable', 'value', by='feature_type', c=feat_type_colors, s=2.5, ax=axs[1]
)
format_ax(axs[1], xticks_size=10, rotx=90)
add_legend(
    label='Feature type',
    colors=feat_type_colors,
    ax=axs[1],
    loc='upper left',
    bbox_to_anchor=(1,1),
    ncols=1,
    label_size=12,
    ticks_size=10
)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_visible(False)
#axs[1].set_yticks([])

# Show
fig.suptitle(f"{n_good}/{df_.shape[0]} ({n_good/df_.shape[0]*100:.2f}%) 'overall good' models (>=0.7 mean f1, over all sample clones)")
plt.show()

fig.savefig(
    os.path.join(path_main, 'images', 'overall_performance_f1.png')
)

##############


##


############## Technical summary

# Plot
colors = {'precision' : '#F0D290', 'recall' : '#DE834D', 'f1' : '#A3423C'}

fig = plt.figure(figsize=(13, 5))
gs = GridSpec(1, 3, figure=fig, width_ratios=[3, 2, 7])

df_ = (
    clones
    .assign(job=lambda x: x['filtering'] + '|' + x['dimred'] + '|' + x['model'] + '|' + x['tuning'])
    .groupby(['sample', 'job'])
    .agg('mean')
    .reset_index(level=1)
)
df_[['filtering', 'dimred', 'model', 'tuning']] = df_['job'].str.split('|', expand=True)
df_ = df_.drop(columns=['job']).reset_index(drop=True)

# Models
ax1 = fig.add_subplot(gs[0,0])
box(
    (
        df_.loc[:, ['f1', 'precision', 'recall', 'model']]
        .melt(id_vars='model', var_name='metric', value_name='score')
    ), 
    'model', 
    'score',
    by='metric', 
    c=colors, 
    hue_order=colors.keys(),
    ax=ax1, 
    kwargs=params
)
format_ax(ax1, xticks_size=10, rotx=90, title='Model', ylabel='value')
add_legend(
    label='Metric',
    colors=colors, 
    ax=ax1,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.3),
    ncols=3,
    label_size=10,
    ticks_size=8
)

ax1.spines[['top', 'right']].set_visible(False)

# Tuning
ax2 = fig.add_subplot(gs[0, 1])
box(
    (
        df_.loc[:, ['f1', 'precision', 'recall', 'tuning']]
        .melt(id_vars='tuning', var_name='metric', value_name='score')
    ), 
    'tuning', 
    'score',
    by='metric', 
    c=colors,
    hue_order=colors.keys(),
    ax=ax2, 
    kwargs=params
)
format_ax(ax2, xticks_size=10, rotx=90, title='Hyperparameters tuning', yticks=[])
ax2.spines[['top', 'right', 'left']].set_visible(False)

# Feature type
ax3 = fig.add_subplot(gs[0, 2])
box(
    (
        df_.loc[:, ['f1', 'precision', 'recall', 'filtering', 'dimred']]
        .assign(feature_type=lambda x: x['filtering'] + '_' + x['dimred'])
        .drop(columns=['dimred', 'filtering'])
        .melt(id_vars='feature_type', var_name='metric', value_name='score')
    ), 
    'feature_type', 
    'score',
    by='metric', 
    c=colors, 
    hue_order=colors.keys(),
    ax=ax3, 
    kwargs=params
)
format_ax(ax3, xticks_size=10, rotx=90, title='Feature matrix', yticks=[])
ax3.spines[['top', 'right', 'left']].set_visible(False)

fig.tight_layout()
plt.show()

##############


##


############## Overall performance by clone

# Summary
df_ = clones.assign(feature_type=lambda x: x['filtering'] + '_' + x['dimred'])
clones_order = (
    df_.groupby('comparison')['f1']
    .agg('mean')
    .sort_values(ascending=False)
    .index
)
n_clones = df_['comparison'].unique().size
good_clones = df_.query('f1>=0.7')['comparison'].unique()
n_good_clones = good_clones.size
df_['clone_status'] = np.where(df_['comparison'].isin(good_clones), '>=1 good models', 'No good models')

# Good clones per sample
(
    df_.loc[:, ['comparison', 'clone_status', 'sample']]
    .drop_duplicates()
    .groupby('sample')
    .apply(lambda x: x['clone_status'].value_counts())
    .reset_index(level=1, name='count')
    .rename(columns={'level_1':'clone_status'})
    .assign(freq=lambda x: 
        x['count'] / x.reset_index().groupby('sample')['count'].transform('sum').values
    )
).to_excel(
    os.path.join(path_main, 'reports', 'clones_status_aggregate_f1.xlsx')
)

##

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)

# Topz
box(df_, 'comparison', 'f1', c='#E9E7E7', ax=axs[0], kwargs=params, order=clones_order)
strip(df_.query('f1>=0.7'), 'comparison', 'f1', c='r', s=3, ax=axs[0], order=clones_order)
strip(df_.query('f1<0.7'), 'comparison', 'f1', c='k', s=1.5, ax=axs[0], order=clones_order)
format_ax(
    axs[0], 
    title=f"{n_good_clones}/{n_clones} ({n_good_clones/n_clones*100:.2f}%) clones with at least 1 'good' model", 
    ylabel='f1 score', 
    rotx=90, 
    xticks_size=5,
    xticks=[]
    )
add_legend(
    label='Model',
    colors={'f1 >= 0.7':'r', 'f1 < 0.7':'k'}, 
    ax=axs[0],
    loc='upper left',
    bbox_to_anchor=(1,1),
    ncols=1,
    label_size=10,
    ticks_size=8
)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[1].spines['bottom'].set_visible(False)

# Bottom:
feat_type_colors = create_palette(df_, 'feature_type', 'Spectral')
box(df_, 'comparison', 'f1', c='#E9E7E7', ax=axs[1], kwargs=params, order=clones_order)
strip(df_, 'comparison', 'f1', by='feature_type', c=feat_type_colors, s=2, ax=axs[1], order=clones_order)
format_ax(
    axs[1],
    ylabel='f1 score', 
    rotx=90,
    xticks_size=5
)
add_legend(
    label='Feature type',
    colors=feat_type_colors, 
    ax=axs[1],
    loc='upper left',
    bbox_to_anchor=(1,1),
    ncols=1,
    label_size=10,
    ticks_size=8
)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Save
fig.tight_layout()
plt.show()
fig.savefig(
    os.path.join(path_main, 'images', 'performance_by_clone_f1.png')
)

##############


##


############## Overall performance by clone 2: relationship with clonal prevalence

fig, axs = plt.subplots(1,3,figsize=(15,5))

# Right
meta = pd.read_csv(os.path.join(path_data, 'cells_meta.csv'), index_col=0)
df_ = (
    meta.assign(clonal=lambda x: x['sample'] + '_' + x['GBC'])
    .groupby('clonal')
    .size()
    .reset_index(name='n')
    .assign(GBC=lambda x: x['clonal'].apply(lambda x: x.split('_')[-1]))
    .assign(sample=lambda x: x['clonal'].apply(lambda x: '_'.join(x.split('_')[:-1])))
    .assign(prevalence=lambda x: x['n'] / x.groupby('sample')['n'].transform('sum'))
    .loc[:, ['GBC', 'prevalence']]
    .set_index('GBC')
)

df_ = (
    clones
    .assign(job=lambda x: x['filtering'] + '_' + x['dimred'] + '_' + x['model'] + '_' + x['tuning'])
    .assign(GBC=lambda x: clones['comparison'].apply(lambda x: x.split('_')[0]))
    .groupby(['GBC', 'job'])
    .agg('mean')
    .reset_index()
    .loc[:, ['GBC', 'f1', 'precision', 'recall']]
    .set_index('GBC')
    .join(df_)
    .sort_values('f1', ascending=False)
)

# Fq
x = df_['prevalence']
y = df_['f1']
corr = np.corrcoef(x, y)[0,1]

axs[0].plot(x, y, 'ko', markersize=2)
sns.regplot(x=x, y=y, ax=axs[0], scatter=False)
format_ax(
    axs[0],
    title=f"Pearson's rho: {corr:.2f}", 
    xlabel='Clonal prevalence', 
    ylabel='f1'
)

x = df_['prevalence']
y = df_['precision']
corr = np.corrcoef(x, y)[0,1]

axs[1].plot(x, y, 'ko', markersize=2)
sns.regplot(x=x, y=y, ax=axs[1], scatter=False)
format_ax(
    axs[1],
    title=f"Pearson's rho: {corr:.2f}", 
    xlabel='Clonal Prevalence', 
    ylabel='Precision'
)

x = df_['prevalence']
y = df_['recall']
corr = np.corrcoef(x, y)[0,1]

axs[2].plot(x, y, 'ko', markersize=2)
sns.regplot(x=x, y=y, ax=axs[2], scatter=False)
format_ax(
    axs[2],
    title=f"Pearson's rho: {corr:.2f}", 
    xlabel='Clonal Prevalence', 
    ylabel='Recall'
)

fig.tight_layout()
plt.show()
############## Overall performance by sample


##


############## Overall performance by sample
fig, axs = plt.subplots(1,3,figsize=(15, 5))

# Left
df_ = (
    clones.loc[:, ['sample','ncells']]
    .drop_duplicates()
    .set_index('sample')
    .loc[['AML_clones', 'MDA_clones', 'MDA_lung', 'MDA_PT']]
    .reset_index()
)

bar(df_, x='sample', y='ncells', c='lightgrey', edgecolor='k', ax=axs[0], s=0.75)
format_ax(axs[0], title='n cells', ylabel='n', xticks=df_['sample'])
axs[0].spines[['top', 'right', 'left']].set_visible(False)

# Center
df_ = (
    clones.loc[:, ['sample','n_clones_analyzed']]
    .drop_duplicates()
    .set_index('sample')
    .loc[['AML_clones', 'MDA_clones', 'MDA_lung', 'MDA_PT']]
    .reset_index()
)

bar(df_, x='sample', y='n_clones_analyzed', c='lightgrey', edgecolor='k', ax=axs[1], s=0.75)
format_ax(axs[1], title='n clones', xticks=df_['sample'])
axs[1].spines[['top', 'right', 'left']].set_visible(False)

# Right
df_ = (
    clones
    .assign(job=lambda x: x['filtering'] + '_' + x['dimred'] + '_' + x['model'] + '_' + x['tuning'])
    .groupby(['sample', 'job'])
    .agg('mean')
    .reset_index()
    .assign(feature_type=lambda x: x['job'].map(lambda x: '_'.join(x.split('_')[:-2])))
    .drop(columns=['job'])
)
samples_order = ['AML_clones', 'MDA_clones', 'MDA_lung', 'MDA_PT']
feat_type_colors = create_palette(df_, 'feature_type', 'Spectral')
box(df_, 'sample', 'f1', c='#E9E7E7', ax=axs[2], kwargs=params, order=samples_order)
strip(df_, 'sample', 'f1', by='feature_type', c=feat_type_colors, s=4, ax=axs[2], order=samples_order)
format_ax(axs[2], title='Classification performance', ylabel='f1 score', xticks_size=10)
add_legend(
    label='Feature type',
    colors=feat_type_colors, 
    ax=axs[2],
    loc='upper left',
    bbox_to_anchor=(1,1),
    ncols=1,
    label_size=10,
    ticks_size=8
)
axs[2].spines[['top', 'right', 'left']].set_visible(False)

# Save
fig.tight_layout()
plt.show()

fig.savefig(
    os.path.join(path_main, 'images', 'performance_by_sample_f1.png')
)

##############


##


############## Overall performance by sample 2: relationship with clonal complexity

fig, axs = plt.subplots(1,3,figsize=(15,5))

# Right
meta = pd.read_csv(os.path.join(path_data, 'cells_meta.csv'), index_col=0)
df_ = (
    meta.groupby(['sample', 'GBC'])
    .size()
    .reset_index(name='n')
    .assign(prevalence=lambda x: x['n'] / x.groupby('sample')['n'].transform('sum'))
)

def SH(df, sample):
    freqs = df.query('sample == @sample')['prevalence'].values
    return -np.sum(freqs * np.log10(freqs))

df_ = pd.Series({ x : SH(df_, x) for x in df_['sample'].unique() }).to_frame('sh')
df_ = (
    clones
    .assign(job=lambda x: x['filtering'] + '_' + x['dimred'] + '_' + x['model'] + '_' + x['tuning'])
    .groupby(['sample', 'job'])
    .agg('mean')
    .reset_index()
    .loc[:, ['sample', 'f1', 'precision', 'recall']]
    .join(df_, on='sample')
    .sort_values('f1')
)

# Fq
x = df_['sh']
y = df_['f1']
corr = np.corrcoef(x, y)[0,1]

axs[0].plot(x, y, 'ko', markersize=4)
sns.regplot(x=x, y=y, ax=axs[0], scatter=False)
format_ax(
    axs[0],
    title=f"Pearson's rho: {corr:.2f}", 
    xlabel='Shannon Entropy', 
    ylabel='f1'
)

x = df_['sh']
y = df_['precision']
corr = np.corrcoef(x, y)[0,1]

axs[1].plot(x, y, 'ko', markersize=4)
sns.regplot(x=x, y=y, ax=axs[1], scatter=False)
format_ax(
    axs[1],
    title=f"Pearson's rho: {corr:.2f}", 
    xlabel='Shannon Entropy', 
    ylabel='Precision'
)

x = df_['sh']
y = df_['recall']
corr = np.corrcoef(x, y)[0,1]

axs[2].plot(x, y, 'ko', markersize=4)
sns.regplot(x=x, y=y, ax=axs[2], scatter=False)
format_ax(
    axs[2],
    title=f"Pearson's rho: {corr:.2f}", 
    xlabel='Shannon Entropy', 
    ylabel='Recall'
)

fig.tight_layout()
plt.show()

##############


##


############## Top 3 models

# Data
df_ = (
    
    clones
    .groupby(['sample', 'comparison'])
    .apply(lambda x: 
        x
        .iloc[:,:5]
        .sort_values('f1', ascending=False)
        .head(3)
    )
    .droplevel(2)
    .sort_values(by='f1', ascending=False)
    .reset_index()
    
)

df_['GBC'] = df_['comparison'].str.split('_').map(lambda x: f'{x[0][:6]}...')

# Vix
samples_order = ['AML_clones', 'MDA_clones', 'MDA_lung', 'MDA_PT']
colors = { s:c for s,c in zip(samples_order, sc.pl.palettes.vega_10) }

# Three good
fig, axs = plt.subplots(1,3,figsize=(11,3.5), sharex=True)
for (ax, sample) in zip(axs, samples_order[:-1]):
    n_clones = df_.query('sample == @sample')['GBC'].unique().size
    strip(
        df_
        .query('sample == @sample'), 
        'GBC',
        'f1',
        by='feature_type',
        c=colors[sample], 
        s=5, 
        ax=ax
    )
    ax.set(ylim=(-0.1,1.1))
    ax.hlines(0.7, 0, n_clones, 'k', 'dashed')
    
    medians = (
        df_.query('sample == @sample')
        .groupby('GBC')['f1']
        .agg('median')
        .sort_values(ascending=False)
    )
    for i, y in enumerate(medians):
        ax.hlines(y, i-.25, i+.25, '#4f4e4a', zorder=2)
    
    format_ax(
        ax, 
        title=sample, 
        xlabel='Ranked clones',
        ylabel='f1',
        rotx=0,
        xticks=[],
        xticks_size=6
    )
    ax.spines[['top', 'right']].set_visible(False)
    
    median_f1 = np.median(df_.query('sample == @sample')['f1'])
    std_f1 = np.std(df_.query('sample == @sample')['f1'])
    median_precision = np.median(df_.query('sample == @sample')['precision'])
    std_precision = np.std(df_.query('sample == @sample')['precision'])
    median_recall = np.median(df_.query('sample == @sample')['recall'])
    std_recall = np.std(df_.query('sample == @sample')['recall'])
    
    ax.text(0.05, 0.71, '0.7', transform=ax.transAxes)
    ax.text(0.05, 0.19, 
        f'precision: {median_precision:.2f} +-{std_precision:.2f}', 
        transform=ax.transAxes
    )
    ax.text(0.05, 0.12, 
        f'recall: {median_recall:.2f} +-{std_recall:.2f}', 
        transform=ax.transAxes
    )
    ax.text(0.05, 0.05, 
        f'f1: {median_f1:.2f} +-{std_f1:.2f}', 
        transform=ax.transAxes
    )

fig.suptitle('Top 3 models')
fig.tight_layout()
plt.show()

##
    
# MDA_PT    
sample = 'MDA_PT'
n_clones = df_.query('sample == @sample')['GBC'].unique().size

fig, ax = plt.subplots(figsize=(11,3))
strip(
    df_
    .query('sample == @sample'), 
    'GBC',
    'f1',
    by='feature_type',
    c=colors[sample], 
    s=5, 
    ax=ax
)
ax.set(ylim=(-0.1,1.1))
ax.hlines(0.7, 0, n_clones, 'k', 'dashed')
format_ax(
    ax, 
    title=f'{sample}',
    xlabel='Ranked clones', 
    ylabel='f1',
    xticks=[],
    rotx=90
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

medians = (
    df_.query('sample == @sample')
    .groupby('GBC')['f1']
    .agg('median')
    .sort_values(ascending=False)
)
for i, y in enumerate(medians):
    ax.hlines(y, i-.25, i+.25, '#4f4e4a', zorder=2)
    
median_f1 = np.median(df_.query('sample == @sample')['f1'])
std_f1 = np.std(df_.query('sample == @sample')['f1'])
median_precision = np.median(df_.query('sample == @sample')['precision'])
std_precision = np.std(df_.query('sample == @sample')['precision'])
median_recall = np.median(df_.query('sample == @sample')['recall'])
std_recall = np.std(df_.query('sample == @sample')['recall'])

ax.text(0.05, 0.71, '0.7', transform=ax.transAxes)
ax.text(0.75, 0.94, 
    f'precision: {median_precision:.2f} +-{std_precision:.2f}', 
    transform=ax.transAxes
)
ax.text(0.75, 0.87, 
    f'recall: {median_recall:.2f} +-{std_recall:.2f}', 
    transform=ax.transAxes
)
ax.text(0.75, 0.80, 
    f'f1: {median_f1:.2f} +-{std_f1:.2f}', 
    transform=ax.transAxes
)

fig.tight_layout()
plt.show()
##############


##


############## 

# Variants biological anno: EDepaquale lab
afm = read_one_sample(path_data, 'AML_clones')
variants = pd.Series(filter_baseline(afm).var_names).to_frame('var')
fun_df = pd.read_csv(path_data + 'rev_table.txt', sep='\t')
fun_df['var'] = fun_df['Position'].astype('str') + \
                '_' + fun_df['Reference'] + \
                '>' + fun_df['Variant'] 
                
            
# Format
df_annot = fun_df[fun_df['var'].isin(variants['var'])]
df_annot = (
    df_annot.loc[:, [
        'Position', 'Reference', 'Variant', 'Consequence', 'Symbol', \
        'Biotype', 'SIFT', 'PolyPhen', 'Disease', 'Status'
    ]]
    .assign(var=lambda x: 
       x['Position'].astype('str') + '_' + x['Reference'] + '>' + x['Variant']
    )
    .drop(columns=['Position', 'Reference', 'Variant'])
    .set_index('var')
)

# Take out positive cells per var info
df_positive = (
    pd.Series(np.sum(afm.X > 0, axis=0), index=afm.var_names)
    .to_frame('n_positive')
    .assign(variant_type=lambda x: x.index.map(lambda x: x.split('_')[1]))
    .loc[lambda x: ~x['variant_type'].str.contains('N>')]
)
order = (
    df_positive.groupby('variant_type')
    .agg('mean')
    .sort_values('n_positive', ascending=False)
    .index
)

# Fig
fig = plt.figure(figsize=(13, 4))
gs = GridSpec(1, 3, figure=fig, width_ratios=[5, 8, 3])

# Subs type
ax = fig.add_subplot(gs[0,0])
strip(df_positive, 'variant_type', 'n_positive', c='#2e2d2d', s=3, order=order, ax=ax)
ax.spines[['top', 'right', 'left']].set_visible(False)
medians = (
    df_positive.groupby('variant_type')
    .agg('mean')
    .sort_values('n_positive', ascending=False)
    .values
)
for i, y in enumerate(medians):
    ax.hlines(y, i-.25, i+.25, 'r', zorder=4)
format_ax(ax, title=f'n of +events, by substitution type', ylabel='n cells', rotx=90)

##

# Gene 
x = 'Symbol'
ax = fig.add_subplot(gs[0,1])
_ = df_annot[x].value_counts().to_frame('count')
bar(_, 'count', c='lightgrey', edgecolor='k', ax=ax)
format_ax(ax, title='MT-SNVs by genomic feature', ylabel='n variants', xticks=_.index.values, 
        rotx=90 if _.index.size>5 else 0)
ax.spines[['top', 'right', 'left']].set_visible(False)

##

# Biotype
x = 'Biotype'
ax = fig.add_subplot(gs[0,2])
_ = df_annot[x].value_counts().to_frame('count')
bar(_, 'count', c='lightgrey', edgecolor='k', ax=ax)
format_ax(ax, title='MT-SNVs by biotype', ylabel='n variants', xticks=_.index.values, rotx=90)
ax.spines[['top', 'right', 'left']].set_visible(False)

# Plot
fig.tight_layout()
plt.show()


##


# Functional anno

# Fig
fig = plt.figure(figsize=(13, 4))
gs = GridSpec(1, 3, figure=fig, width_ratios=[7, 5, 5])

# Codon
_ = df_annot['Consequence'].value_counts().to_frame('count')
_.index = [
    'missense',
    'synonimous',
    'non_coding',
    'stop_gain',
    'incomplete_terminal',
    'intergenic',
    'stop_retained'
]

ax = fig.add_subplot(gs[0,0])
bar(_, 'count', c='lightgrey', edgecolor='k', ax=ax)
format_ax(ax, title='MT-SNVs codon effect', ylabel='n variants', xticks=_.index.values, rotx=90)
ax.spines[['top', 'right', 'left']].set_visible(False)

# SIFT
_ = df_annot['SIFT'].map(lambda x: x.split('(')[0]).value_counts().to_frame('count')
ax = fig.add_subplot(gs[0,1])
bar(_, 'count', c='lightgrey', edgecolor='k', ax=ax)
format_ax(ax, title='Protein effect (SIFT)', ylabel='n variants', xticks=_.index.values, rotx=90)
ax.spines[['top', 'right', 'left']].set_visible(False)

# Polyphen
_ = df_annot['PolyPhen'].map(lambda x: x.split('(')[0]).value_counts().to_frame('count')
ax = fig.add_subplot(gs[0,2])
bar(_, 'count', c='lightgrey', edgecolor='k', ax=ax)
format_ax(ax, title='Protein effect (PolyPhen)', ylabel='n variants', xticks=_.index.values, rotx=90)
ax.spines[['top', 'right', 'left']].set_visible(False)

# Plot
fig.tight_layout()
plt.show()

##

##############


##


############## 

# Are homoplasmic, high-prevalence variants under active selection??
var_prevalence = np.sum(afm.X>0, axis=0) / afm.shape[0]
var_mean_AF = np.nanmean(afm.X, axis=0)
df_ = (
    pd.DataFrame(
        {'prevalence' : var_prevalence, 'mean_AF': var_mean_AF},
        index=afm.var_names
    )
    .assign(type=lambda x: np.where((x['prevalence']>0.5) & (x['mean_AF']>0.5), 'high', 'low'))
    .join(df_annot.loc[:, ['Consequence', 'SIFT', 'PolyPhen']], how='inner') 
    # NB. Make sure df_annot is filtered!
)
n_high = df_.query('type == "high"').shape[0]

##

# Plot
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, figure=fig, width_ratios=[2,2])

# Biplot mean_AF/prevalence
ax = fig.add_subplot(gs[0,0])
ax.plot(df_.query('type == "low"')['mean_AF'], df_.query('type == "low"')['prevalence'], 'k.', markersize=3)
ax.plot(df_.query('type == "high"')['mean_AF'], df_.query('type == "high"')['prevalence'], 'r+', markersize=6)
format_ax(ax, xlabel='Mean AF', ylabel='Prevalence')

ax.text(.4, .2, f'MT-SNVs:', transform=ax.transAxes)
ax.text(.4, .15, f'-{df_.shape[0]} passing baseline filters', transform=ax.transAxes)
ax.text(.4, .1, f'-{n_high} >0.5 mean_AF and prevalence',  transform=ax.transAxes)
ax.spines[['right', 'top']].set_visible(False)

# Lollipop consequence
ax = fig.add_subplot(gs[0,1])
stem_plot(df_.query('type == "high"').sort_values('mean_AF', ascending=False), 'mean_AF', ax=ax)
format_ax(ax, xlabel='Mean AF')
ax.spines[['right', 'top', 'left']].set_visible(False)

tests = [
    df_['Consequence'] == 'non_coding_transcript_exon_variant',
    df_['Consequence'] == 'synonymous_variant',
    df_['Consequence'] == 'missense_variant'
]
df_['codon'] = np.select(tests, ['non coding', 'synonimous', 'missense'])
annot_list = df_.query('type == "high"').loc[:, ['codon', 'PolyPhen']].values.tolist()

_ = .97
i = 0
vep, poly = annot_list[i]
#print(i, vep, poly)
ax.text(.1, _, f'VEP: {vep}; PolyPhen: {poly}', transform=ax.transAxes)
for x in annot_list[1:]:
    i += 1
    vep, poly = x
    #print(i, vep, poly)
    ax.text(.1, _-(0.1*i), f'VEP: {vep}; PolyPhen: {poly}', transform=ax.transAxes)

fig.tight_layout()
plt.show()

##############


##


############## 