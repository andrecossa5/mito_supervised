"""
Script to visualize GT variants.
"""

import os
import warnings
from mito_utils.preprocessing import *
from mito_utils.clustering import *
from mito_utils.dimred import *
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.plotting_base import *
from mito_utils.diagnostic_plots import *
from mito_utils.embeddings_plots import *
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('macOSX')


##


# Args
path_main = '/Users/IEO5505/Desktop/mito_bench'


##

# Set paths
path_data = os.path.join(path_main, 'data') 
path_output = os.path.join(path_main, 'results', 'supervised_clones', 'output')
path_viz = os.path.join(
    path_main, 'results', 'supervised_clones', 'visualization', 'variant_subsets_and_GT'
)
path_tmp = os.path.join(
    path_main, 'results', 'supervised_clones', 'downstream_files', 'variant_subsets_and_GT'
)


##


# Load data
afm = read_one_sample(path_data, sample, with_GBC=True)
with open(os.path.join(path_tmp, f'{sample}_GT_variants.pickle'), 'rb') as f:
    d = pickle.load(f)

# Filter and umap
a_cells, a = filter_cells_and_vars(afm, variants=d['(0.75,0.25)'])
a = nans_as_zeros(a)
X_umap, name_dims = reduce_dimensions(a, 'UMAP', metric='cosine', n_comps=2)
df_ = (
    pd.DataFrame(X_umap, columns=name_dims, index=a.obs_names)
    .join(pd.DataFrame(a.X, index=a.obs_names, columns=a.var_names))
    .join(a.obs)
)

# Viz
df_.groupby('GBC').size().sort_values()
rank_clone_variants(a, group='CCCGCTGCGGCTGCTCCG')


# Viz
var = '9624_T>C'
clone = 'CCCGCTGCGGCTGCTCCG'

fig, axs = plt.subplots(1,3, figsize=(13.5,4.5))

draw_embeddings(
    df_, cont=var, ax=axs[0], 
    title=f'{clone}: {var}',
    cbar_kwargs={'vmin':.01, 'vmax':.1}
)
plot_exclusive_variant(a_cells, var, ax=axs[1])
df_clone = (
    df_.loc[:,[var, 'GBC']]
    .assign(cat=lambda x: np.where(x['GBC'] == clone, clone, 'other'))
)
box(df_clone, x='cat', y=var, c='white', ax=axs[2])
strip(df_clone, x='cat', y=var, c='k', s=2.5, ax=axs[2])
format_ax(axs[2], ylabel='AF', title=f'{var} AF', reduced_spines=True)

fig.tight_layout()
fig.savefig(
    os.path.join(path_viz, f'{sample}_{var}.png'),
    dpi=300
)


##


# Viz one distribution
fig, ax = plt.subplots(figsize=(5,4.5))
hist(df_, var, sturges(df_[var]), c='k', ax=ax)
format_ax(ax, title=f'{var} AF', reduced_spines=True, xlabel='AF', ylabel='ncells')
fig.savefig(
    os.path.join(path_viz, f'{sample}_{var}_hist.png'), 
    dpi=300
)


##