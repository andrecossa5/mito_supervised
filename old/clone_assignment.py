"""
Assign cells to lentiviral clones: create GT reference.
"""

import os
import sys
import pickle
import pandas as pd
from itertools import chain
import numpy as np
sys.path.append('/Users/IEO5505/Desktop/MI_TO/mito_preprocessing/bin/sc_gbc')
from helpers import *


##


# Args
path_input_invivo = '/Users/IEO5505/Desktop/BC_chemo_reproducibility/data/MDA'
path_out = '/Users/IEO5505/Desktop/mito_bench/data/'
path_clones = os.path.join(path_input_invivo, 'clonal_info')


##


# Read MDA data, in vivo longitudinal samples 
meta = pd.read_csv(os.path.join(path_input_invivo, 'cells_meta_orig.csv'), index_col=0)


##


def process_one_sample(
    name_in_vivo='NT_NT_PTs_2',
    sample='MDA_PT',
    correction_type='reference-free',
    coverage_treshold=15,
    umi_treshold=5,
    p_treshold=.1,
    ratio_to_most_abundant_treshold=.3
    ):

    print(f'Go sample {name_in_vivo}/{sample}')

    # Read counts as pickle
    with open(os.path.join(
            path_input_invivo, 
            'clonal_info', 
            f'{name_in_vivo}_counts.pickle'), 'rb') as p:
        COUNTS = pickle.load(p)

    # Filter only QCed cells
    counts = COUNTS[correction_type]
    counts = counts.loc[
        counts['CBC']
        .isin(meta.query('sample==@name_in_vivo')
        .index.map(lambda x: x.split('_')[0]))
    ].copy()

    # Filter UMIs
    counts = mark_UMIs(counts, coverage_treshold=coverage_treshold)

    # Viz filtering
    fig, ax = plt.subplots(figsize=(5,5))
    viz_UMIs(counts, by='status', ax=ax, nbins=50)
    fig.tight_layout()
    fig.savefig(os.path.join(path_out, sample, 'selected_UMIs.png'), dpi=300)

    ##

    # Find clones
    df_combos = get_combos(counts, gbc_col=f'GBC_{correction_type}')
    M, _ = filter_and_pivot(
        df_combos, 
        umi_treshold=umi_treshold, 
        p_treshold=p_treshold, 
        ratio_to_most_abundant_treshold=ratio_to_most_abundant_treshold
    )
    sets = get_clones(M)
    GBC_set = list(chain.from_iterable(sets['GBC_set'].map(lambda x: x.split(';')).to_list()))
    redundancy = 1-np.unique(GBC_set).size/len(GBC_set)
    occurrences = pd.Series(GBC_set).value_counts().sort_values(ascending=False)
    print(f'- Unique GBCs sets: {sets.shape[0]}')
    print(f'- Unique GBCs in these sets: {np.unique(GBC_set).size}')
    print(f'- GBCs redundancy across sets: {redundancy:.2f}')
    print(f'- GBCs occurrences across sets: {occurrences.median():.2f} (+- {occurrences.std():.2f})')

    # Top clones GBC co-occurrence:
    for x in sets['GBC_set'][:25].values:
        print(sets.loc[sets['GBC_set'].str.contains(x)]['n cells'].to_dict())

    # Assign cells
    unique_cells = (M>0).sum(axis=1).loc[lambda x: x==1].index
    filtered_M = M.loc[unique_cells]
    clones_df = get_clones(filtered_M)
    cells_df = (
        filtered_M
        .apply(lambda x: filtered_M.columns[x>0][0], axis=1)
        .to_frame('GBC')
    )
    print(f'# Final clones (i.e., distinct populations of uniquely barcoded cells only) checks')
    print(f'- n starting CBC (STARSolo): {df_combos["CBC"].unique().size}')
    print(f'- n uniquely barcoded cells: {cells_df.shape[0]}')
    print(f'- n clones: {clones_df.shape[0]}')
    print(f'- n clones>=10 cells: {clones_df["n cells"].loc[lambda x:x>=10].size}')

    # Save
    clones_df.to_csv(os.path.join(path_out, sample, 'clones_summary_table.csv'))
    cells_df.to_csv(os.path.join(path_out, sample, 'cells_summary_table.csv'))
    (
        cells_df.index.to_frame()
        .to_csv(os.path.join(path_out, sample, 'barcodes.txt'), index=False)
    )


##


# Sample specific
    
# MDA_PT
process_one_sample(
    name_in_vivo='NT_NT_PTs_2',
    sample='MDA_PT',
    correction_type='reference-free',
    coverage_treshold=15,
    umi_treshold=5,
    p_treshold=.5,
    ratio_to_most_abundant_treshold=.3
)


##


# MDA_lung
process_one_sample(
    name_in_vivo='NT_NT_mets_2',
    sample='MDA_lung',
    correction_type='reference-free',
    coverage_treshold=50,
    umi_treshold=5,
    p_treshold=.5,
    ratio_to_most_abundant_treshold=.3
)


##