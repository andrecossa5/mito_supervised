#!/usr/bin/python

import os
from itertools import product

combos = list(product(['MDA_clones', 'MDA_clones_old'], ['stringent', 'enriched']))

for sample, filtering in combos:
    os.system(f'python multi_classification.py {sample} {filtering} 0.05')
