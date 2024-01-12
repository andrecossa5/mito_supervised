"""
Script to filter variants within an AFM.
"""

import os
import warnings
from mito_utils.preprocessing import *
from mito_utils.utils import *
from mito_utils.preprocessing import *
warnings.filterwarnings('ignore')


##


# Args
# sample = 'AML_clones'
# path_main = '/Users/IEO5505/Desktop/mito_bench'

path_main = sys.argv[1]
sample = sys.argv[2]


##

# Set paths
path_data = os.path.join(path_main, 'data') 
path_tmp = os.path.join(
    path_main, 'results', 'supervised_clones', 'downstream_files', 'variant_subsets_and_GT'
)


# Filtering options
filtering_options = ['miller2022', 'ludwig2019', 'pegasus', 'MQuad']


##


def main():

    # Load data
    afm = read_one_sample(path_data, sample, with_GBC=True)

    # Here we go
    d = {}
    for x in filtering_options:
        filtered_vars = filter_cells_and_vars(
            afm, filtering=x, nproc=8, path_=path_tmp, n=100
        )[1].var_names
        d[x] = filtered_vars.to_list()

    # Save
    with open(os.path.join(path_tmp, f'{sample}_filtered_subsets.pickle'), 'wb') as f:
        pickle.dump(d, f)

##

###################################################

# Run 
if __name__ == '__main__':
    main()
