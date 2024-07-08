#!/usr/bin/python

"""
Investigation of MT-SNVs metrics spaces.
1. What is the best performing metric (i.e., the one that achieve better AUCPR on gt clonal labels)?
2. What kNN graph (fixed k, varying metric) has the best association with gt clonal labels?
3. What is the relationship among all differnt metrics, considering the resulting kNN graphs?
"""

import os
import sys
import pickle
from mito_utils.preprocessing import *
from mito_utils.clustering import *
from mito_utils.utils import *
from mito_utils.distances import *
from mito_utils.kNN import *
from mito_utils.metrics import *
from mito_utils.plotting_base import *


##


# Set paths
# path_main = '/Users/IEO5505/Desktop/mito_bench'
# sample = 'AML_clones'
# filtering = 'enriched'
# t = 0.05

path_main = sys.argv[1]
sample = sys.argv[2]
filtering = sys.argv[3]
t = float(sys.argv[4])

path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results/supervised_clones/distances')
path_vars = os.path.join(path_main, 'results/var_selection')


##


def main():

    # Data
    make_folder(path_results, sample, overwrite=False)
    path_sample = os.path.join(path_results, sample)
    afm = read_one_sample(path_data, sample, with_GBC=True)

    # Filter AFM
    with open(os.path.join(path_vars, 'variants.pickle'), 'rb') as f:
        VARIANTS = pickle.load(f)

    _, a = filter_cells_and_vars(
        afm, variants=VARIANTS[(sample, filtering)], 
        max_AD_counts=2, af_confident_detection=0.05, min_cell_number=10
    )
    labels = a.obs['GBC']

    ##

    ############################## 
    # 1. What is the best performing metric 
    # (i.e., the one that achieve better AUCPR on gt clonal labels)?
    ############################## 

    # Here we go
    results = {}
    metrics = [ 'cosine', 'correlation', 'sqeuclidean', 'jaccard', 'hamming']
    n_samples = 10
    n_cells_sampling = round((a.shape[0] / 100) * 80)

    results = {}
    for metric in metrics:

        l = []
        for _ in range(n_samples):

            cells_ = np.random.choice(a.obs_names, size=n_cells_sampling, replace=False)
            a_ = a[cells_,:]
            a_.uns['per_position_coverage'] = a_.uns['per_position_coverage'].loc[cells_,:]
            labels_ = a_.obs['GBC']
            l.append(evaluate_metric_with_gt(a_, metric, labels_, t=t))

        results[metric] = l

    # Save
    results_path = os.path.join(path_sample, f'evaluation_metrics_aucpr_{filtering}.pickle')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    ##

    ############################## 
    # 2. What kNN graph (fixed k, varying metric) has the
    #  best association with gt clonal labels (kBET metric)?
    ############################## 

    # Here we go
    results = {}

    # Compute kNNs and corresponding idx
    k = 15
    labels = a.obs['GBC'].astype('str')

    for metric in metrics:

        idx = kNN_graph(a.X, k=k, nn_kwargs={'metric':metric})[0]
        _, _, acc_rate = kbet(idx, labels, alpha=t, only_score=False)
        median_entropy = NN_entropy(idx, labels)
        median_purity = NN_purity(idx, labels)
        results[metric] = {
            'kBET_rejection_rate' : 1-acc_rate, 
            'median_NN_entropy': median_entropy, 
            'median_NN_purity' : median_purity
        }

    ##

    # Save
    results_path = os.path.join(path_sample, f'evaluation_metrics_kNN_{filtering}.pickle')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    ############################## 
    # 3. What metric is more robust to noise??
    ############################## 

    # Here we go
    results = {}
    n_samples = 100
    n_vars_sampling = round((a.shape[1] / 100) * 80)

    results = {}
    for metric in metrics:
        L = []
        for _ in range(n_samples):
            vars_ = np.random.choice(a.var_names, size=n_vars_sampling, replace=False)
            D = pair_d(a[:,vars_].X, metric=metric, t=t, ncores=8)
            L.append(D.flatten())
        results[metric] = np.mean(np.corrcoef(np.array(L)))

    # Save
    results_path = os.path.join(path_sample, f'evaluation_metrics_corr_{filtering}.pickle')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)


    ##

#####################################################

# Run
if __name__ == "__main__":
    main()


