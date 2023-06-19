#!/usr/bin/python

import os
from itertools import product

# Path
path_wd = '/Users/IEO5505/Desktop/MI_TO/mito_supervised/downstream'
# Combos
samples = ['AML_clones', 'MDA_clones', 'MDA_lung', 'MDA_PT']
filterings = ['pegasus', 'miller2022', 'MQuad']
combos = product(samples, filterings)

# Launch
os.chdir(path_wd)
for s, f in combos:
    print(f'Running {s}, {f}...')
    ncov = 100 if s != 'MDA_PT' else 10
    os.system(f'python distance_metrics_evaluation.py {s} {f} {ncov}')