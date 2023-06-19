"""
Visualization of a single-clone with low-prevalence and high performance.
"""

# Code
import sys
import os
import re
import pickle
from scipy.stats import pearsonr
from mito_utils.preprocessing import *
from mito_utils.clustering import *
from mito_utils.dimred import *
from mito_utils.embeddings_plots import *
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

# Load classification report
clones = pd.read_csv(os.path.join(path_report, 'report_f1.csv'), index_col=0)


##


# Find nice example
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
    .sort_values(by='AUCPR', ascending=False)
    .join(
        clones.loc[:, 
            ['clone_prevalence', 'sample', 'comparison']
        ].set_index(['sample', 'comparison']), 
    )
    .reset_index()
    .drop_duplicates()
)

df_.query('AUCPR > 0.8 and clone_prevalence <0.1').sort_values('clone_prevalence')

# MDA_clones, CCGCAATAAATGCGACTT_vs_rest here
sample = 'MDA_PT'
clone = 'GCTATCACATTTAAGTTG'
query = '''
    AUCPR > 0.8 and clone_prevalence <0.1 and sample == "MDA_PT" \
    and comparison == "GCTATCACATTTAAGTTG_vs_rest"
'''
clone_d = clones.query(query).T.iloc[:,0].to_dict()

# Get variants
path_pickle = os.path.join(path_report, f'../output/out_MDA_PT_MQuad_no_dimred_xgboost_random_10.pickle')
with open(path_pickle, 'rb') as f:
    d = pickle.load(f)
variants = d['trained_models']['AAGCACAAATCGCTAGGG_vs_rest']['variants']

# Read AFM
afm = read_one_sample(path_data, sample=sample, with_GBC=True)

# Filter
_, a = filter_cells_and_vars(
    afm,
    sample=sample,
    min_cell_number=clone_d['min_cell_number'], 
    min_cov_treshold=clone_d['min_cov_treshold'],
    variants=variants
)

# Prep input for phylo 
# (
#     pd.DataFrame(a.X, index=a.obs_names, columns=a.var_names)
#     .to_csv(os.path.join(path_viz, 'AML_clones_afm.csv'))
# )
# a.obs.to_csv(os.path.join(path_viz, 'AML_clones_meta.csv'))

##

# Find exclusive variants
vois_df = rank_clone_variants(
    a, clone=clone, 
    min_clone_perc=0.4, 
    max_perc_rest=0.1
)
vois = vois_df.index.to_list()

############### SEED 9, NOVE. Nine. ###############

# Visualize variants
a = nans_as_zeros(a)
colors = create_palette(a.obs, 'GBC', sc.pl.palettes.godsnot_102)

# for seed in range(2, 10):
#     X, _ = reduce_dimensions(a, method='UMAP', n_comps=2, sqrt=False, seed=i)

X, _ = reduce_dimensions(a, method='UMAP', n_comps=2, sqrt=False, seed=9)
embs = (
    pd.DataFrame(X, columns=['UMAP1', 'UMAP2'], index=a.obs_names)
    .join(
        [
            a.obs,
            pd.DataFrame(a[:, vois].X.copy(), columns=vois, index=a.obs_names)
        ]
    )
)

# Viz clone UMAP
fig, ax = plt.subplots(figsize=(5,5))
c_ = embs.query('GBC == @clone')
o_ = embs.query('GBC != @clone')
ax.plot(o_['UMAP1'], o_['UMAP2'], '.', c='#9A9588', markersize=1)
ax.plot(c_['UMAP1'], c_['UMAP2'], '.', c=colors[clone], markersize=5)
fig.savefig(os.path.join(path_viz, f'var'))
format_ax(
    ax, 
    title=f'{clone} clone ({sample})', 
    xlabel='UMAP1 (MT-SNVs)', 
    ylabel='UMAP2 (MT-SNVs)', 
    xticks=[], yticks=[]
)
ax.text(.52, .15, f'n cells: {clone_d["ncells_clone"]} (sample: {clone_d["ncells_sample"]})', transform=ax.transAxes)
ax.text(.52, .1, f'Prevalence: {clone_d["clone_prevalence"]:.2e}', transform=ax.transAxes)
ax.text(.52, .05, f'AUCPR: {clone_d["AUCPR"]:.2e}', transform=ax.transAxes)
fig.savefig(os.path.join(path_viz, 'the_gem.png'), dpi=500)


##


# Viz variants
fig, ax = plt.subplots(figsize=(7,3.5))

df_= a.obs.join(pd.DataFrame(a[:, vois].X, columns=vois, index=a.obs_names))
df_['group'] = np.where(df_['GBC']==clone, clone[:5], 'others')
df_ = df_.loc[:, ['group', *vois]].melt(id_vars=['group'])

var_colors = create_palette(df_, 'variable', 'tab10')#, lightness=1, saturation=1)

box(df_, x='group', y='value', by='variable', c=var_colors, ax=ax)
format_ax(ax, title='Highly exclusive MT-SNVs', ylabel='AF')
ax.spines[['right', 'top']].set_visible(False)

ax.text(.05, .88, 
    f'9097_A>G, % +clone, +rest: {vois_df.loc["9097_A>G", "perc_clone"]:.2f}, {vois_df.loc["9097_A>G", "perc_rest"]:.2f}', 
    transform=ax.transAxes, fontdict={'color':var_colors["9097_A>G"], 'fontweight': 'bold'}
)
ax.text(.05, .81, 
    f'9321_T>C, % +clone, +rest: {vois_df.loc["9321_T>C", "perc_clone"]:.2f}, {vois_df.loc["9321_T>C", "perc_rest"]:.2f}', 
    transform=ax.transAxes, fontdict={'color':var_colors["9321_T>C"], 'fontweight': 'bold'}
)
ax.text(.05, .74, 
    f'7027_C>T, % +clone, +rest: {vois_df.loc["7027_C>T", "perc_clone"]:.2f}, {vois_df.loc["7027_C>T", "perc_rest"]:.2f}', 
    transform=ax.transAxes, fontdict={'color':var_colors["7027_C>T"], 'fontweight': 'bold'}
)

fig.tight_layout()
fig.savefig(os.path.join(path_viz, f'{clone}_top_variants.png'), dpi=1000)


##


# Heatmap
g = cells_vars_heatmap(a, cmap='mako', cell_anno='GBC', anno_colors=colors, heat_label=None, 
    legend_label='Clones', figsize=(11, 10), title=None, cbar_position=(0.82, 0.2, 0.02, 0.25),
    title_hjust=0.47, legend_bbox_to_anchor=(0.825, 0.5), legend_loc='lower center', 
    legend_ncol=1, xticks_size=5, order='diagonal_clones', vmax=0.025
)
g.fig.savefig(os.path.join(path_viz, f'{clone}_heatmap.png'))


##

















