"""
UMAP samples from GT variants.
"""

import os
from mito_utils.preprocessing import *
from mito_utils.dimred import *
from mito_utils.utils import *
from mito_utils.plotting_base import *
from mito_utils.embeddings_plots import *
from mito_utils.kNN import kNN_graph
from mito_utils.metrics import NN_purity
from mito_utils.distances import pair_d
matplotlib.use('macOSX')


##


# Args
path_main = '/Users/IEO5505/Desktop/mito_bench'

##


# Set paths
path_data = os.path.join(path_main, 'data') 
path_vars = os.path.join(path_main, 'results', 'var_selection')
path_results = os.path.join(path_main, 'results', 'supervised_clones')


##


# Load GT variants and clone colors
with open(os.path.join(path_vars, 'variants.pickle'), 'rb') as f:
    d = pickle.load(f)
# Colors
with open(os.path.join(path_data, 'clones_colors_sc.pickle'), 'rb') as f:
    colors = pickle.load(f)


##

# jaccard, .05, k=5, stringent --> good
# jaccard, .05, k=5, stringent --> good

# Compute UMAPs
UMAPs = {}
purities = {}
for sample in ['MDA_clones', 'AML_clones']:
    afm = read_one_sample(path_data, sample, with_GBC=True)
    _, a = filter_cells_and_vars(afm, variants=d[(sample, 'MQuad')], max_AD_counts=2, af_confident_detection=0.05, min_cell_number=10)
    labels = a.obs['GBC'].astype('str')
    D = pair_d(a.X, metric='jaccard', t=.05, ncores=8)
    idx = kNN_graph(D, k=5, from_distances=True)[0]
    purities[sample] = NN_purity(idx, labels)
    X_umap, name_dims = reduce_dimensions(a, 'UMAP', metric='jaccard', k=35, n_comps=2, seed=1234)
    umap = (
        pd.DataFrame(X_umap, columns=name_dims, index=a.obs_names)
        .join(pd.DataFrame(a.X, index=a.obs_names, columns=a.var_names))
        .join(a.obs)
    )
    UMAPs[sample] = umap


##


# from mito_utils.clustering import *
# 
# 
# # Utils
# def place_anno(s, d_colors, order, orientation='horizontal', pos=(0, 1, 1, 0.1), ax=None):
#     colors = s.astype('str').map(d_colors).values[order]
#     axins = ax.inset_axes(pos) 
#     cmap = matplotlib.colors.ListedColormap(colors)
#     cb = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=axins, orientation=orientation)
#     cb.ax.set(xticks=[], yticks=[])
# 
# order = leaves_list(linkage(D))
# d_colors = { k:colors[k] for k in colors if k in a.obs['GBC'].unique() }
# 
# fig, ax = plt.subplots(figsize=(5,5))
# ax.imshow(D[np.ix_(order, order)], cmap='viridis_r')
# place_anno(a.obs['GBC'], d_colors, order, orientation='horizontal', pos=(0, 1, 1, 0.07), ax=ax)
# plt.show()




# Viz clones
s = 8
fig, axs = plt.subplots(2,2,figsize=(5.5,6))

sample = 'AML_clones'
draw_embeddings(
    UMAPs[sample], 
    # 'Diff1', 'Diff2',
    # 'PC1', 'PC2',
    cat='GBC', ax=axs[0,0], 
    title=f'{sample} (kNN purity={purities[sample]:.2f})',
    legend_kwargs={'colors': { k:colors[k] for k in colors if k in UMAPs[sample]['GBC'].unique() } },
    axes_kwargs={'legend':False},
    s=s
)
axs[0,0].axis('off')

sample = 'MDA_clones'
draw_embeddings(
    UMAPs[sample], 
    # 'Diff1', 'Diff2',
    # 'PC1', 'PC2',
    cat='GBC', ax=axs[0,1], 
    title=f'{sample} (kNN purity={purities[sample]:.2f})',
    legend_kwargs={'colors': { k:colors[k] for k in colors if k in UMAPs[sample]['GBC'].unique() } },
    axes_kwargs={'legend':False}, 
    s=s
)
axs[0,1].axis('off')

sample = 'MDA_PT'
draw_embeddings(
    UMAPs[sample], 
    # 'Diff1', 'Diff2',
    # 'PC1', 'PC2',
    cat='GBC', ax=axs[1,0], 
    title=f'{sample} (kNN purity={purities[sample]:.2f})',
    legend_kwargs={'colors': { k:colors[k] for k in colors if k in UMAPs[sample]['GBC'].unique() } },
    axes_kwargs={'legend':False}, 
    s=s
)
axs[1,0].axis('off')

sample = 'MDA_lung'
draw_embeddings(
    UMAPs[sample],
    # 'Diff1', 'Diff2',
    # 'PC1', 'PC2',
    cat='GBC', ax=axs[1,1], 
    title=f'{sample} (kNN purity={purities[sample]:.2f})',
    legend_kwargs={'colors': { k:colors[k] for k in colors if k in UMAPs[sample]['GBC'].unique() } },
    axes_kwargs={'legend':False}, 
    s=s
)
axs[1,1].axis('off')

fig.tight_layout()
plt.show()

fig.savefig(
    os.path.join(path_results, f'GBC_GT_umaps.pdf'),
    # os.path.join(path_results, f'PCA.png'),
    dpi=500
)


##