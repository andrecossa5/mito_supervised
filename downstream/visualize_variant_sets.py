"""
Visualize different sets of variants.
"""

import os
import pickle
import pandas as pd
from mito_utils.preprocessing import *
from mito_utils.distances import pair_d
from mito_utils.dimred import *
from mito_utils.kNN import *
from mito_utils.clustering import *
from mito_utils.embeddings_plots import *
matplotlib.use('macOSX')


##


# Utils
def place_anno(s, d_colors, order, orientation='horizontal', pos=(0, 1, 1, 0.04), ax=None):
    colors = s.astype('str').map(d_colors).values[order]
    axins = ax.inset_axes(pos) 
    cmap = matplotlib.colors.ListedColormap(colors)
    cb = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=axins, orientation=orientation)
    cb.ax.set(xticks=[], yticks=[])

##


# Paths
path_main = '/Users/IEO5505/Desktop/mito_bench/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'var_selection')

# Filtering
sample = 'MDA_clones'
afm = read_one_sample(path_data=path_data, sample=sample, with_GBC=True)

# Here we go
D_VARS = {}
D_METRICS = {}
r = np.arange(5,50,10)

for af_conf, n_conf in list(product(r,r)):

    key = f'n{n_conf}>{af_conf}'
    d = {
        'min_site_cov' : 25, 
        'min_var_quality' : 30, 
        'min_frac_negative' : 0,
        'min_n_positive' : 0,
        'af_confident_detection' : af_conf/100,
        'min_n_confidently_detected' : n_conf
    }
    dataset_df, a = filter_cells_and_vars(
        afm, filtering='MI_TO', min_cell_number=10, filtering_kwargs=d, lineage_column='GBC'
    )
    d = dataset_df.set_index('metric')['value'].to_dict()
    n_cells = d['n_cells']
    n_vars = d['n_vars']
    n_transitions = a.var_names.str.contains('|'.join(transitions)).sum()
    n_transversions = a.var_names.str.contains('|'.join(transversions)).sum()
    n_biased = a.var.loc[:,a.var.columns.str.contains('enr')].apply(lambda x: x.any(), axis=1).sum()
    metrics = {'n_conf':n_conf,'af_conf': af_conf, 'n_cells':n_cells, 'n_vars':n_vars,
               'n_transitions':n_transitions, 'n_transversions':n_transversions, 'n_biased':n_biased}

    D_VARS[key] = a.var_names.tolist()
    D_METRICS[key] = metrics


##


# Metrics
metrics = pd.DataFrame(D_METRICS).T

# Viz
df_ = metrics[['n_conf', 'af_conf', 'n_cells', 'n_vars', 'n_biased']]

# n muts and n enriched, for each MT-SNVs set
df_ = df_.sort_values('n_vars', ascending=False)
order = df_.index
df_ = df_.reset_index().rename(columns={'index':'set'})

fig, axs = plt.subplots(2,1,figsize=(7, 4),sharex=True)
bar(df_, x='set', y='n_vars', ax=axs[0], edgecolor='k', s=.75)
format_ax(axs[0], xticks=order, rotx=90, ylabel='n variants')
axs[0].spines[['right', 'top', 'left']].set_visible(False)
bar(df_, x='set', y='n_biased', ax=axs[1], edgecolor='k', s=.75)
format_ax(axs[1], xticks=order, rotx=90, xlabel='MT-SNVs set', ylabel='n enriched')
axs[1].spines[['right', 'top', 'left']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'{sample}_n_vars_n_enriched.png'), dpi=1000)


##


# Tresholding effect and underlying clonal structures

# Load colors
with open(os.path.join(path_data, 'clones_colors_sc.pickle'), 'rb') as f:
    D = pickle.load(f)
colors = { k:D[k] for k in D if k in a.obs['GBC'].unique() }

# Viz
chosen = ['n5>5', 'n5>15', 'n5>25', 'n5>35', 'n5>45']

_, a = filter_cells_and_vars(afm, variants=D_VARS[chosen[-1]], min_cell_number=10)
order_cosine = leaves_list(linkage(pair_d(a, metric='cosine'), method='weighted'))
order_jaccard = leaves_list(linkage(pair_d(a), method='weighted'))

# Fig
fig, axs = plt.subplots(2,5,figsize=(15,6.3))

for i, key in enumerate(chosen):
    _, a = filter_cells_and_vars(afm, variants=D_VARS[key], min_cell_number=10)
    d = pair_d(a)
    axs[0,i].imshow(d[np.ix_(order_jaccard, order_jaccard)], cmap='viridis_r')
    place_anno(a.obs['GBC'], colors, order_jaccard, ax=axs[0,i])
    axs[0,i].set(xticks=[], yticks=[], title=f'{key} (n={a.shape[1]}) ', xlabel='Cells')
    d = pair_d(a, metric='cosine')
    axs[1,i].imshow(d[np.ix_(order_cosine, order_cosine)], cmap='viridis_r')
    place_anno(a.obs['GBC'], colors, order_cosine, ax=axs[1,i])
    axs[1,i].set(xticks=[], yticks=[], xlabel='Cells')
    # X, cols = reduce_dimensions(a, method='UMAP', metric='cosine', n_comps=2, seed=1)
    # df_ = pd.DataFrame(X, columns=cols, index=a.obs_names).join(a.obs)
    # draw_embeddings(df_, cat='GBC', ax=axs[1,i], title='', 
    #                 legend_kwargs={'colors':colors}, axes_kwargs={'legend':False})
fig.subplots_adjust(wspace=.2, hspace=.2)
fig.savefig(os.path.join(path_results, f'{sample}_var_sets.png'), dpi=1000)


##


# POLISH!!!!


# sns.kdeplot(calc_median_AF_in_positives(a, a.var_names[a.var_names.str.contains('|'.join(transitions))]))
# sns.kdeplot(calc_median_AF_in_positives(a, a.var_names[a.var_names.str.contains('|'.join(transversions))]))
# 
# sns.kdeplot(a.var.loc[a.var_names.str.contains('|'.join(transitions))]['quality'])
# sns.kdeplot(a.var.loc[a.var_names.str.contains('|'.join(transversions))]['quality'])
# plt.show()


