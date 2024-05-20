"""
Longitudinal couple variants
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
path_results = os.path.join(path_main, 'results', 'longitudinal')

# Create clonal_small.csv
samples = [ s for s in os.listdir(path_data) if s in ["MDA_PT", "MDA_lung"] ]
AFMs = { s : read_one_sample(path_data, sample=s) for s in samples }

x = AFMs['MDA_lung'].obs['GBC'].astype('str')
x.value_counts().size
-np.sum(x.value_counts(normalize=True)*np.log10(x.value_counts(normalize=True)))


# GT variants
a_lung, _ = filter_afm_with_gt(AFMs['MDA_lung'])
a_PT, _ = filter_afm_with_gt(AFMs['MDA_PT'])
# summary_stats_vars(a_lung)
# summary_stats_vars(a_PT)

# Select clones and variants
common_GT = list(set(a_lung.var_names) & set(a_PT.var_names))
long_clones = list(
    set(a_lung.obs['GBC'].unique()) & set(a_PT.obs['GBC'].unique())
)

# Compute AFs
cells = a_PT.obs.query('GBC in @long_clones').index
df_PT = (
    pd.DataFrame(
        a_PT[cells, common_GT].X, 
        index=cells, columns=common_GT
    )
    .join(a_PT[cells,:].obs)
)
cells = a_lung.obs.query('GBC in @long_clones').index
df_lung = (
    pd.DataFrame(
        a_lung[cells, common_GT].X, 
        index=cells, columns=common_GT
    )
    .join(a_lung[cells,:].obs)
)
df = pd.concat([df_lung, df_PT])
df_AF = (
    df.reset_index(drop=True)
    .melt(id_vars=['sample', 'GBC'], var_name='mut', value_name='AF')
)

# Reorder muts
cat = df.iloc[:,:9].mean(axis=0).sort_values(ascending=False)
df_AF['mut'] = pd.Categorical(df_AF['mut'], categories=cat.index.to_list())


##


# Viz

# Read clone colors
with open(os.path.join(path_data, 'clones_colors_sc.pickle'), 'rb') as f:
    clones_colors = pickle.load(f)
# Filter
clones_colors = { k:clones_colors[k] for k in long_clones }

# Fig
fig, axs = plt.subplots(2,1,figsize=(7.5,5), sharex=True)
box(df_AF.query('sample=="MDA_PT"'), x='mut', y='AF', by='GBC', c=clones_colors, ax=axs[0])
box(df_AF.query('sample=="MDA_lung"'), x='mut', y='AF', by='GBC', c=clones_colors, ax=axs[1])
format_ax(axs[0], title='PT', ylabel='AF', reduced_spines=True)
format_ax(axs[1], title='lung', ylabel='AF', reduced_spines=True, rotx=90)
add_legend(
    'Clones', colors=clones_colors, ax=axs[0], bbox_to_anchor=(1,1), loc='upper right',
    ticks_size=8, label_size=10, artists_size=9
)
fig.tight_layout()
fig.savefig(os.path.join(path_results, 'muts_AF.pdf'), dpi=500)

# Difference PT-lung
df_diff = (
    df.groupby(['sample', 'GBC'])
    .median().reset_index()
    .melt(id_vars=['sample', 'GBC'], var_name='mut', value_name='AF')
    .set_index(['sample', 'GBC', 'mut'])
    .join(
        (
            df.groupby(['sample', 'GBC'])
            .apply(lambda x: (x>0).sum()/x.shape[0])
            .reset_index()
            .melt(id_vars=['sample', 'GBC'], var_name='mut', value_name='%')
            .set_index(['sample', 'GBC', 'mut'])
        )
    )
    .reset_index()
)


# Grouped box
# fig, ax = plt.subplots(figsize=(5,5))


##
    

# Met potential and muts
df_freq = (
    df[['GBC', 'sample']]
    .groupby('sample')
    ['GBC'].value_counts(normalize=True)
    .reset_index(name='freq')
    .query('sample in ["MDA_PT", "MDA_lung"]')
)

# Expansions
df_exp = (
    df_freq
    .pivot_table(index='GBC', columns=['sample'], values='freq')
    .dropna()
    .assign(
        met_potential=lambda x: x['MDA_lung']/x['MDA_PT'],
    )
)

df_diff.query('GBC=="GTCGCTGTCCTGCTCCCG"').sort_values('%', ascending=False)








