"""
Summary of classification performances.
"""

# Code
import re
import os
import sys
import numpy as np
import pandas as pd

# Set paths
path_main = sys.argv[1]
path_clones = path_main + '/results_and_plots/clones_classification/'
path_samples = path_main + '/results_and_plots/samples_classification/'


##


# Clones
res = []
for x in os.listdir(path_clones):
    if x.endswith('.xlsx') and not bool(re.search('report', x)):
        run_pars = x.split('_')[1:-1]
        df = pd.read_excel(path_clones + x, index_col=0
            ).reset_index().loc[:, 
                ['feature_type', 'comparison', 'evidence']
            ].drop_duplicates().assign(
                sample=run_pars[0], min_cell_number=int(run_pars[2]), 
                min_cov_treshold=int(run_pars[3]), model=run_pars[-1], analysis='_'.join(run_pars[:-1])
            ).rename(columns={'evidence':'f1'}).loc[:,
                [ 
                    'analysis', 'sample', 'feature_type', 'min_cov_treshold', 
                    'min_cell_number', 'model', 'comparison', 'f1' 
                ]
            ]
        res.append(df)

# Write clones
res = pd.concat(res, axis=0).sort_values('f1', ascending=False)
res.to_excel(path_clones + 'report_classification_clones.xlsx')


##


# Samples
res = []
for x in os.listdir(path_samples):
    if x.endswith('.xlsx') and not bool(re.search('report', x)):
        run_pars = x.split('_')[1:-1]
        df = pd.read_excel(path_samples + x, index_col=0
            ).reset_index().loc[:, 
                ['feature_type', 'comparison', 'evidence']
            ].drop_duplicates().assign( 
                min_cov_treshold=int(run_pars[1]), model=run_pars[-1], analysis='_'.join(run_pars[:-1])
            ).rename(columns={'evidence':'f1'}).loc[:,
                [ 'analysis', 'feature_type', 'min_cov_treshold', 'model', 'comparison', 'f1' ]
            ]
        res.append(df)

# Write samples
res = pd.concat(res, axis=0).sort_values('f1', ascending=False)
res.to_excel(path_samples + 'report_classification_samples.xlsx')

    


