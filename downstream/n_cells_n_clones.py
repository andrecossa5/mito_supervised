"""
n cells and clones by sample
"""

import os
from mito_utils.plotting_base import *
matplotlib.use('macOSX')


##

# Paths
path_main = '/Users/IEO5505/Desktop/mito_bench/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'sample_diagnostics')

# Read clonal_small.csv
df = pd.read_csv(os.path.join(path_data, 'clonal_small.csv'), index_col=0)

# Viz
fig, axs = plt.subplots(1,2,figsize=(6,4.5))

df_ = df.groupby('sample').size().to_frame('n cells').reset_index().sort_values('n cells')
bar(df_, x='sample', y='n cells', edgecolor='k', c='k', s=.7, a=.7, ax=axs[0])
format_ax(axs[0], ylabel='n cells', xticks=df_['sample'], reduced_spines=True, rotx=90)

df_ = df.groupby('sample')['GBC'].nunique().to_frame('n clones').reset_index().sort_values('n clones')
bar(df_, x='sample', y='n clones', edgecolor='k', c='k', s=.7, a=.7, ax=axs[1])
format_ax(axs[1], ylabel='n clones', xticks=df_['sample'], reduced_spines=True, rotx=90)

fig.tight_layout()
fig.savefig(os.path.join(path_results, 'ncells_nclones.png'), dpi=300)


##