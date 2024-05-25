"""
Clones identifiability.
"""

import os
import pandas as pd
from mito_utils.preprocessing import *
from mito_utils.plotting_base import *
matplotlib.use('macOSX')


##


# Paths
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data') 
path_results = os.path.join(path_main, 'results', 'var_selection')

#

# Read clones_df
df = pd.read_csv(os.path.join(path_results, 'clones_df', 'clones_df.csv'), index_col=0)
df['sample'] = pd.Categorical(df['sample'], categories=['AML_clones', 'MDA_clones', 'MDA_lung', 'MDA_PT'][::-1])

# Viz detectable vs undetectable clones
df_enriched = (
    df.query('filtering=="enriched"')
    .assign(clone_status=lambda x: x['id_lineage'].map({True:'detectable', False:'non-detectable'}))
)
df_stringent = (
    df.query('filtering=="stringent"')
    .assign(clone_status=lambda x: x['id_lineage'].map({True:'detectable', False:'non-detectable'}))
)


##


#  Fig
clone_colors = {'detectable':'#DA590F', 'non-detectable':'#0EA25D'}
sample_colors = create_palette(df_stringent, 'sample', ten_godisnot)

fig, axs = plt.subplots(2,2,figsize=(8,4.5), sharex=True)
bb_plot(df_stringent, 'sample', 'clone_status', colors=clone_colors, ax=axs[0], legend=False)
axs[0].set(title='At least one MT-SNV with: \n +cells clone >= 75%, +cells rest <= 25%, median AF >= 5%', xlabel='')
add_legend(label='Clone status', colors=clone_colors, ax=axs[0], loc='upper left', bbox_to_anchor=(1,1), ticks_size=9, label_size=10, artists_size=10)
bb_plot(df_enriched, 'sample', 'clone_status', colors=clone_colors, ax=axs[1], legend=False)
axs[1].set(title='At least one enriched MT-SNV (Fisher\'s test, p<.05)', xlabel='% clones in sample')

fig.tight_layout()
fig.savefig(os.path.join(path_results, 'clone_status.png'), dpi=500)


##


# n vars
fig, axs = plt.subplots(2,1,figsize=(8,4.5), sharex=True)

for sample in sample_colors:
    sns.kdeplot(df_stringent.query('sample==@sample')['n_vars'], color=sample_colors[sample], fill=True, alpha=.4, ax=axs[0])
axs[0].set(title='n MT-SNV with: \n +cells clone >= 75%, +cells rest <= 25%, median AF >= 5%', xlabel='n MT-SNVs')
axs[0].axvline(df_stringent['n_vars'].median(), c='k', linestyle='--')
axs[0].spines[['right', 'top']].set_visible(False)
add_legend('Sample', colors=sample_colors, ax=axs[0], loc='upper left', bbox_to_anchor=(1,1), ticks_size=9, label_size=10, artists_size=10)

for sample in sample_colors:
    sns.kdeplot(df_enriched.query('sample==@sample')['n_vars'], color=sample_colors[sample], fill=True, alpha=.4, ax=axs[1])
axs[1].set(title='n MT-SNV (Fisher\'s test, p<.05)', xlabel='n MT-SNVs')
axs[1].spines[['right', 'top']].set_visible(False)
axs[1].axvline(df_enriched['n_vars'].median(), c='k', linestyle='--')

fig.tight_layout()
fig.savefig(os.path.join(path_results, 'n_vars.png'), dpi=500)


##