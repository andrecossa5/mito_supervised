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
path_results = os.path.join(path_main, 'results', 'supervised_clones')


##


# Load GT variants and clone colors
with open(os.path.join(path_results, 'variants.pickle'), 'rb') as f:
    d = pickle.load(f)
# Extract only GT
GT = { k[0] : d[k] for k in d if k[1]=='GT' }
# Colors
with open(os.path.join(path_data, 'clones_colors_sc.pickle'), 'rb') as f:
    colors = pickle.load(f)



# Compute UMAPs
UMAPs = {}
purities = {}
for sample in GT:
    afm = read_one_sample(path_data, sample, with_GBC=True)
    a_cells, a = filter_cells_and_vars(afm, variants=GT[sample], min_cell_number=10)
    a = nans_as_zeros(a)
    labels = a.obs['GBC'].astype('str')
    idx = kNN_graph(a.X, k=30, nn_kwargs={'metric':'cosine'})[0]
    purities[sample] = NN_purity(idx, labels)
    X_umap, name_dims = reduce_dimensions(a, 'UMAP', metric='cosine', n_comps=2, seed=1)
    umap = (
        pd.DataFrame(X_umap, columns=name_dims, index=a.obs_names)
        .join(pd.DataFrame(a.X, index=a.obs_names, columns=a.var_names))
        .join(a.obs)
    )
    UMAPs[sample] = umap


##


# Viz clones
s = 8
fig, axs = plt.subplots(2,2,figsize=(5.5,6))

sample = 'AML_clones'
draw_embeddings(
    UMAPs[sample], cat='GBC', ax=axs[0,0], 
    title=f'{sample} (kNN purity={purities[sample]:.2f})',
    legend_kwargs={'colors': { k:colors[k] for k in colors if k in UMAPs[sample]['GBC'].unique() } },
    axes_kwargs={'legend':False},
    s=s
)
# axs[0,0].axis('off')

sample = 'MDA_clones'
draw_embeddings(
    UMAPs[sample], cat='GBC', ax=axs[0,1], 
    title=f'{sample} (kNN purity={purities[sample]:.2f})',
    legend_kwargs={'colors': { k:colors[k] for k in colors if k in UMAPs[sample]['GBC'].unique() } },
    axes_kwargs={'legend':False}, 
    s=s
)
# axs[0,1].axis('off')

sample = 'MDA_PT'
draw_embeddings(
    UMAPs[sample], cat='GBC', ax=axs[1,0], 
    title=f'{sample} (kNN purity={purities[sample]:.2f})',
    legend_kwargs={'colors': { k:colors[k] for k in colors if k in UMAPs[sample]['GBC'].unique() } },
    axes_kwargs={'legend':False}, 
    s=s
)
# axs[1,0].axis('off')

sample = 'MDA_lung'
draw_embeddings(
    UMAPs[sample], cat='GBC', ax=axs[1,1], 
    title=f'{sample} (kNN purity={purities[sample]:.2f})',
    legend_kwargs={'colors': { k:colors[k] for k in colors if k in UMAPs[sample]['GBC'].unique() } },
    axes_kwargs={'legend':False}, 
    s=s
)
# axs[1,1].axis('off')

fig.tight_layout()
fig.savefig(
    os.path.join(path_results, f'GBC_GT_umaps.png'),
    dpi=500
)


##