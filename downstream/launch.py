#!/usr/bin/python

import os
from itertools import product

combos = list(product(['MDA_clones', 'AML_clones'], ['stringent', 'enriched']))

for sample, filtering in combos:
    os.system(f'python multi_classification.py {sample} {filtering} 0.05')