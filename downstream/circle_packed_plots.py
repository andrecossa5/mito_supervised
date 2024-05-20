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

# Create clonal_small.csv
# samples = [ s for s in os.listdir(path_data) if bool(re.search('AML|MDA', s)) ]
# AFMs = { s : read_one_sample(path_data, sample=s) for s in samples }
# df = pd.concat([ AFMs[k].obs for k in AFMs ])
# df.to_csv(os.path.join(path_data, 'clonal_small.csv'))

# Read clonal_small.csv
df = pd.read_csv(os.path.join(path_data, 'clonal_small.csv'), index_col=0)

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
fig = plt.figure(figsize=(15, 3.5))

order = ['AML_clones', 'MDA_clones', 'MDA_PT', 'MDA_lung']
for i, sample in enumerate(order):

    ax = fig.add_subplot(1, 4, i+1)
    df_ = df_freq.query('sample==@sample and freq>=0.01').set_index('GBC')
    packed_circle_plot(
        df_, covariate='freq', ax=ax, color=clones_colors, annotate=True, t_cov=.05,
        alpha=.65, linewidth=1.5, fontsize=8, fontcolor='k', fontweight='medium'
# fontweight or weight: {a numeric value in range 0-1000, 'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'de
# mibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'}
    )
    ax.set(title=sample)
    fig.tight_layout()
    ax.axis('off')

fig.subplots_adjust(bottom=.1, top=.9, left=.1, right=.9, wspace=.2, hspace=.5)
fig.savefig(os.path.join(path_results, 'circle_packed.pdf'), dpi=1000)


##