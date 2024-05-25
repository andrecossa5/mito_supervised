"""
Trial filter_cells_and_vars.
"""

import os
from mito_utils.preprocessing import *


##


# Paths
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data') 
path_results = os.path.join(path_main, 'results', 'supervised_clones')


##


# Read data
sample = 'MDA_clones'
min_cell_number = 10



afm = read_one_sample(path_data, sample, with_GBC=True)

_, a = filter_cells_and_vars(
    afm, filtering='MI_TO', max_AD_counts=10, fit_mixtures=True, only_positive_deltaBIC=True
)

a.var_names.map(lambda x: x.split('_')[1] in transitions).values.sum()
a.var_names.map(lambda x: x.split('_')[1] in transversions).values.sum()





