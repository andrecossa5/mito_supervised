"""
Circle plot longitudinal couple
"""

import os
import re
import random
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
    

# Met potential
df_freq = (
    df[['GBC', 'sample']]
    .groupby('sample')
    ['GBC'].value_counts(normalize=True)
    .reset_index(name='freq')
    .query('sample in ["MDA_PT", "MDA_lung"]')
)

# Expansions
df = (
    df_freq
    .pivot_table(index='GBC', columns=['sample'], values='freq')
    .dropna()
    .assign(
        met_potential=lambda x: x['MDA_lung']/x['MDA_PT'],
    )
)

fig, ax = plt.subplots(figsize=(5,4.5))
scatter(df, 'MDA_PT', 'MDA_lung', s=100, by='met_potential', c='Spectral_r', ax=ax)
format_ax(
    title=f'Clonal expansions, PT-lung (n={df.shape[0]})',
    xlabel='PT prevalence',
    ylabel='lung prevalence',
    ax=ax
)
add_cbar(
    df['met_potential'], palette='Spectral_r', ax=ax, label='met_potential', layout='outside'
)
ax.spines[['right', 'top']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(path_results, 'met_potential_couple.pdf'), dpi=500)


##


# Fig 
fig = plt.figure(figsize=(6.7, 3.5))

ax = fig.add_subplot(1, 2, 1)
packed_circle_plot(
    df, covariate='MDA_PT', ax=ax, color='met_potential', 
    annotate=True, t_cov=.01, alpha=.7, linewidth=1.5, cmap='Spectral_r'
)
ax.set(title='PT')
ax.axis('off')

ax = fig.add_subplot(1, 2, 2)
packed_circle_plot(
    df, covariate='MDA_lung', ax=ax, color='met_potential', 
    annotate=True, t_cov=.01, alpha=.7, linewidth=1.5, cmap='Spectral_r'
)
ax.set(title='lung')
ax.axis('off')

# Save
fig.tight_layout()
fig.savefig(os.path.join(path_results, 'circle_packed.pdf'), dpi=500)


##