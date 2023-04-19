#!/usr/bin/python

import pandas as pd
from itertools import product

# Lists
samples = ['MDA_clones', 'AML_clones']
input_mode = ['more_stringent']
filtering = ['miller2022', 'MQuad', 'ludwig2019', 'pegasus']
dimred = ['no_dimred', 'PCA', 'UMAP', 'diffmap'] 
models = [ 'logit', 'kNN', 'xgboost']
min_cell_number = [10]

# Product, and write
jobs = list(product(samples, input_mode, filtering, dimred, models, min_cell_number)) 
pd.DataFrame(
    jobs, 
    columns=['sample', 'input_mode', 'filtering', 'dimred', 'model', 'min_cell_number']
).to_csv('jobs.csv')



