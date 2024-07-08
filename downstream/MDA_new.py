"""
Is it better?
"""

import os
import pickle
from mito_utils.preprocessing import *
from mito_utils.clustering import *
from mito_utils.utils import *
from mito_utils.dimred import *
from mito_utils.distances import *
from mito_utils.kNN import *
from mito_utils.metrics import *
from mito_utils.plotting_base import *
from mito_utils.embeddings_plots import *
from mito_utils.phylo_plots import *
from mito_utils.phylo import *


##


# Set paths
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results/supervised_clones/distances')
path_vars = os.path.join(path_main, 'results/var_selection')

# Params
sample = 'MDA_clones_old'
filtering = 'GT_enriched'
t = 0.05

# Data
make_folder(path_results, sample, overwrite=False)
path_sample = os.path.join(path_results, sample)
afm = read_one_sample(path_data, sample, with_GBC=True, nmads=10)

afm.obs_names

# Read
_, a = filter_cells_and_vars(
    afm, filtering=filtering, lineage_column='GBC',
    max_AD_counts=2, af_confident_detection=0.05, min_cell_number=5
)
labels = a.obs['GBC']
a

# Distances
X, dimnames = reduce_dimensions(a, method='UMAP', metric='euclidean', n_comps=2, seed=1)
df_ = pd.DataFrame(X,columns=dimnames, index=a.obs_names).join(a.obs)


fig, ax = plt.subplots(figsize=(6,4.5))
draw_embeddings(df_, cat='GBC', ax=ax, s=25, legend_kwargs={'loc':'upper left'})
fig.tight_layout()
plt.show()


# Tree
tree = build_tree(a, solver='NJ', t=.05)
tree.cell_meta['GBC'] = tree.cell_meta['GBC'].astype('str')

fig, ax = plt.subplots(figsize=(5,5))
plot_tree(tree, ax=ax, meta=['GBC'])
fig.tight_layout()
plt.show()

calculate_corr_distances(tree)


##


mito_utils