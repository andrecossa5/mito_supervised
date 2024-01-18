"""
Last QC on trascriptional covariates to remove technical artifacts 
or spurios subpopulations.
"""

import os
import numpy as np
import pandas as pd
from itertools import chain
from plotting_utils._plotting_base import *
from mito_utils.preprocessing import *
matplotlib.use('macOSX')


##

# Paths
path_main = '/Users/IEO5505/Desktop/mito_bench/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'sample_diagnostics')

# Read MT data in a dictionary
samples = [ s for s in os.listdir(path_data) if bool(re.search('AML|MDA', s)) ]
good_cells_d = { s : pd.read_csv(os.path.join(path_data, s, 'barcodes.txt'))['CBC'].to_list() for s in samples }
good_cells_l = list(chain.from_iterable([ good_cells_d[k] for k in good_cells_d ]))

# Read all cells metadata from first transcriptional QC
meta = pd.read_csv(os.path.join(path_data, 'cells_meta_orig.csv'), index_col=0)
# Only good, barcoded cells
meta = meta.loc[good_cells_l]

# Explore covariates
# sample = 'MDA_lung'
# fig, ax = plt.subplots(figsize=(5,5))
# sns.kdeplot(data=meta.query('sample==@sample'), x='nUMIs', y='doublet_score', ax=ax)
# fig.tight_layout()
# plt.show()

# Filter out last cells according to doublet score and nUMIs
params = {
    'AML_clones' : [0.15, 85000], 
    'MDA_clones' : [0.15, 150000], 
    'MDA_PT' : [0.15, 30000], 
    'MDA_lung' : [0.15, 60000], 
}

# Viz and save final cells
final_cells = {}
fig, axs = plt.subplots(1,4,figsize=(15.5,4))

for i,sample in enumerate(good_cells_d):
    sns.kdeplot(data=meta.query('sample==@sample'), x='nUMIs', y='doublet_score', ax=axs[i])
    dscore = params[sample][0]
    nUMIs = params[sample][1]
    axs[i].axvline(x=nUMIs, c='k')
    axs[i].axhline(y=dscore, c='k')
    final_cells[sample] = (
        meta
        .query('sample==@sample and doublet_score<=@dscore and nUMIs<=@nUMIs')
        .index.to_list()
    )
    axs[i].set(title=f'{sample}: final cells {len(final_cells[sample])}')

fig.tight_layout()
fig.savefig(os.path.join(path_main, 'results', 'GT_clonal_assignment', 'final_QC.png'), dpi=300)


##


# Save barcodes
for k in final_cells:
    (
        pd.DataFrame({'CBC':final_cells[k]})
        .to_csv(
            os.path.join(path_data, k, 'barcodes.txt'), index=False
        )
    )


##




