#!/usr/bin/python

import os
from itertools import product

# Path
path_wd = '/Users/IEO5505/Desktop/MI_TO/mito_supervised/downstream'

# Combos
samples = ['MDA_PT', 'MDA_lung']
filterings = ["MQuad", "pegasus", "miller2022", "ludwig2019", "GT"]
combos = product(samples, filterings)

# Launch
os.chdir(path_wd)
for s, f in combos:
    print(f'Running {s}, {f}...')
    os.system(f'python multi_classification.py {s} {f}')