"""
Exploratory analysis and checks on the three original and formatted AFMs.
"""

import os
import re
import random
from matplotlib.colors import Normalize
from mito_utils.preprocessing import *
from mito_utils.diagnostic_plots import *
from mito_utils.plotting_base import *
matplotlib.use('macOSX')


##

# Paths
path_main = '/Users/IEO5505/Desktop/mito_bench/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'sample_diagnostics')

# Read cells_meta.csv
df = pd.read_csv(os.path.join(path_data, 'cells_meta.csv'), index_col=0)

# Packed circles clones single-cell
df_freq = (
    df[['GBC', 'sample']]
    .groupby('sample')
    ['GBC'].value_counts(normalize=True)
    .reset_index(name='freq')
)

# Colors, for each clone sample
df_freq['GBC'].unique().size


# Random colors for clones
# clones = df_freq['GBC'].unique()
# random.seed(125)
# clones_colors = { 
#     clone : color for clone, color in \
#     zip(
#         clones, 
#         list(
#             ''.join( ['#'] + [random.choice('ABCDEF0123456789') for i in range(6)] )  \
#             for _ in range(clones.size)
#         )
#     )
# }
# with open(os.path.join(path_data, 'clones_colors_sc.pickle'), 'wb') as f:
#     pickle.dump(clones_colors, f)

# Read colors
with open(os.path.join(path_data, 'clones_colors_sc.pickle'), 'rb') as f:
    clones_colors = pickle.load(f)


##


# Fig
order = ['AML_clones', 'MDA_clones', 'MDA_PT', 'MDA_lung']

fig, axs = plt.subplots(2,2,figsize=(7,7))
for ax, sample in zip(axs.flat, order):
    f = .01 if sample in ['MDA_PT', 'MDA_lung'] else .001
    df_ = df_freq.query('sample==@sample and freq>=@f').set_index('GBC')
    packed_circle_plot(
        df_, covariate='freq', ax=ax, color=clones_colors, annotate=True, t_cov=.05,
        alpha=.65, linewidth=1.5, fontsize=7.5, fontcolor='k', fontweight='medium'
    )
    ax.set(title=sample)
    ax.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(path_results, 'circle_packed.png'), dpi=1000)


##