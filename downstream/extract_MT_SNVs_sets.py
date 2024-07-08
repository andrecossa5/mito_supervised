"""
Extract MT-SNVs pickle from binary classification output.
"""

# Code
import os
import re
import pickle
import pandas as pd
import numpy as np


##

# Args
path_results = '/Users/IEO5505/Desktop/mito_bench/results/supervised_clones/'


##


# Extraxt MT-SNVs subsets
d = {}
pickles = [ 
    x for x in os.listdir(os.path.join(path_results, 'outs')) \
    if bool(re.search('.pickle', x)) 
]
for p in pickles:
    sample = '_'.join(p.split('_')[1:3])
    filtering_key = '_'.join(p.split('_')[3:-5])  
    print(sample, filtering_key)
    with open(os.path.join(path_results, 'outs', p), 'rb') as f:
        r = pickle.load(f)
    k = r['performance_df'][['sample', 'filtering_key']].drop_duplicates().values[0]
    k = tuple(k)
    if k not in d:
        _ = list(r['trained_models'].keys())[0]
        d[(sample, filtering_key)] = r['trained_models'][_]['variants'].to_list()


# Save
with open(os.path.join(path_results, 'variants.pickle'), 'wb') as f:
    pickle.dump(d, f)


##