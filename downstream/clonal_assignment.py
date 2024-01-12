"""
Visual inspection script to find appropriate, sample-specific coverage tresholds to remove
ambient RNA contamination and filter good quality UMIs.
"""

import os
import sys
import pickle
import scanpy as sc
import pandas as pd
import numpy as np
from itertools import chain
from plotting_utils._plotting_base import *
sys.path.append('/Users/IEO5505/Desktop/MI_TO/mito_preprocessing/bin/sc_gbc')
from helpers import *


##


# Args
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'GT_clonal_assignment')


##


# Read meta e AFM
meta = pd.read_csv(os.path.join(path_data, 'cells_meta_orig.csv'), index_col=0)


##


# Single sample exploratory 

# Check single-samples
sample = 'MDA_PT'

# Params
correction_type = 'reference'
coverage_treshold = 25
umi_treshold = 5
p_treshold = 1
max_ratio_treshold = .5
normalized_abundance_treshold = .5


##


# Read AFM and lentiviral counts
afm = sc.read(os.path.join(path_data, sample, 'AFM.h5ad'))
with open(os.path.join(path_data, sample, 'counts.pickle'), 'rb') as p:
    COUNTS = pickle.load(p)

# Filter only QCed cells
counts = COUNTS[correction_type]
counts = counts.loc[
    counts['CBC']
    .isin(meta.query('sample==@sample')
    .index.map(lambda x: x.split('_')[0]))
].copy()


##


# Filter UMIs
counts = mark_UMIs(counts, coverage_treshold=coverage_treshold, nbins=50)
fig, ax = plt.subplots(figsize=(5,5))
viz_UMIs(counts, by='status', ax=ax, nbins=50)
ax.set(title=sample)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'{sample}_filtered_UMIs.png'), dpi=300)

# Get combos
df_combos = get_combos(counts, gbc_col=f'GBC_{correction_type}')

# Filtering CBC-GBC
M, _ = filter_and_pivot(
    df_combos, 
    umi_treshold=umi_treshold, 
    p_treshold=p_treshold,  
    max_ratio_treshold=max_ratio_treshold,
    normalized_abundance_treshold=normalized_abundance_treshold
)

# GBC sets checks
sets = get_clones(M)
sets.head(20)
GBC_set = list(chain.from_iterable(sets['GBC_set'].map(lambda x: x.split(';')).to_list()))
redundancy = 1-np.unique(GBC_set).size/len(GBC_set)
occurrences = pd.Series(GBC_set).value_counts().sort_values(ascending=False)
occurrences.median()

# Get 1-GBC CBCs
unique_cells = (M>0).sum(axis=1).loc[lambda x: x==1].index

# Merge with AFM and change names
afm.obs_names = afm.obs_names.map(lambda x: '_'.join([x, sample]))
unique_cells = unique_cells.map(lambda x: '_'.join([x, sample]))
common = set(meta.query("sample==@sample").index) & set(unique_cells) & set(afm.obs_names)
common = list(common)

# Get final clones
M.index = M.index.map(lambda x: '_'.join([x, sample]))
filtered_M = M.loc[unique_cells]
clones_df = get_clones(filtered_M)
cells_df = (
    filtered_M
    .apply(lambda x: filtered_M.columns[x>0][0], axis=1)
    .to_frame('GBC_set')
)

# Final clones checks
print(f'# Final clones (i.e., distinct populations of uniquely barcoded cells only) checks \n')
print(f'- n starting CBC (STARSolo): {COUNTS[correction_type]["CBC"].unique().size}\n')
print(f'- n starting CBC (QC cells): {meta.query("sample==@sample").shape[0]}\n')
print(f'- n uniquely barcoded cells with MITO: {len(common)}\n')
print(f'- n final clones: {clones_df.shape[0]}\n')
print(f'- n clones>=10 cells: {clones_df["n cells"].loc[lambda x:x>=10].size}\n')

# Viz p_poisson vs nUMIs
fig, ax = plt.subplots(figsize=(5,5))
scatter(df_combos, 'umi', 'p', by='max_ratio', marker='o', s=10, vmin=.2, vmax=.8, ax=ax, c='Spectral_r')
format_ax(
    ax, title='p Poisson vs nUMIs, all CBC-GBC combinations', 
    xlabel='nUMIs', ylabel='p', reduce_spines=True
)
ax.axhline(y=p_treshold, color='k', linestyle='--')
ax.text(.2, .9, f'Total CBC-GBC combo: {df_combos.shape[0]}', transform=ax.transAxes)
n_filtered = df_combos.query('status=="supported"').shape[0]
ax.text(.2, .86, 
    f'n CBC-GBC combo retained: {n_filtered} ({n_filtered/df_combos.shape[0]*100:.2f}%)',
    transform=ax.transAxes
)
ax.set(title=sample)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'{sample}_UMIs_vs_poisson.png'), dpi=300)


##


# Viz final CBC-GBC combos
fig, axs = plt.subplots(1,2,figsize=(10,5))
sns.scatterplot(data=df_combos, x='normalized_abundance', y='max_ratio', ax=axs[0])
axs[0].axvline(x=normalized_abundance_treshold, color='k')
axs[0].axhline(y=max_ratio_treshold, color='k')
sns.kdeplot(data=df_combos, x='normalized_abundance', y='max_ratio', ax=axs[1])
axs[1].axvline(x=normalized_abundance_treshold, color='k')
axs[1].axhline(y=max_ratio_treshold, color='k')
axs[1].text(.27,.13,
    f'n CBC-GBC combinations: {np.sum(df_combos["status"]=="supported")} ({np.sum(df_combos["status"]=="supported")/df_combos.shape[0]*100:.2f}%)',
    transform=axs[1].transAxes
)
axs[1].text(.27,.09, f'n 1-GBC cells: {cells_df.shape[0]}', transform=axs[1].transAxes)
fig.suptitle(sample)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'{sample}_selected_CBC_GBC.png'), dpi=300)


##


# Save
cells_df.to_csv(os.path.join(path_data, sample, 'cells_summary_table.csv'))
clones_df.to_csv(os.path.join(path_data, sample, 'clones_summary_table.csv'))