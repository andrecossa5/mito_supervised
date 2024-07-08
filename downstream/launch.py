#!/usr/bin/python

import os
import json
from itertools import product
from mito_utils.utils import *


path_main = '/data/cossa2_dare/MI_TO_benchmark/'
path_filtering = '/data/cossa2_dare/MI_TO_benchmark/data/filtering_options.json'
samples = ['MDA_clones', 'AML_clones', 'MDA_PT', 'MDA_lung']
with open(path_filtering, 'r') as file:
    FILTERING_OPTIONS = json.load(file)
filtering_keys = FILTERING_OPTIONS.keys()

# Run
combos = list(product(samples, filtering_keys))
for sample, filtering in combos:
    os.system(f'python multi_classification.py {path_main} {sample} {filtering} 0.05')
