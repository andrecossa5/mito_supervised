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
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data')
path_supervised = os.path.join(path_main, 'results/supervised_clones')
path_distances = os.path.join(path_supervised, 'distances')

# Params
# sample = 'MDA_clones'
# filtering = 'MQuad'

sample = sys.argv[1]
filtering = sys.argv[2]


##


def main():

    # Data
    make_folder(path_distances, sample, overwrite=False)
    path_sample = os.path.join(path_distances, sample)
    afm = read_one_sample(path_data, sample, with_GBC=True)

    # Filter AFM
    with open(os.path.join(path_supervised, 'variants.pickle'), 'rb') as f:
        VARIANTS = pickle.load(f)
    _, a = filter_cells_and_vars(
        afm, sample=sample, variants=VARIANTS[(sample, filtering)],
    )
    a = nans_as_zeros(a)
    labels = a.obs['GBC']

    ##

    ############################## 
    # 1. What is the best performing metric 
    # (i.e., the one that achieve better AUCPR on gt clonal labels)?
    ############################## 

    # Here we go
    results = {}
    metrics = [
        'euclidean', 'sqeuclidean', 'cosine', 
        'correlation', 'jaccard', 'matching', 'ludwig2019'
    ]
    n_samples = 5
    n_cells_sampling = round((a.shape[0] / 100) * 80)

    results = {}
    for metric in metrics:

        l = []
        for _ in range(n_samples):
            cells_ = np.random.choice(a.obs_names, size=n_cells_sampling, replace=False)
            a_ = a[cells_,:]
            a_.uns['per_position_coverage'] = a_.uns['per_position_coverage'].loc[cells_,:]
            labels_ = a_.obs['GBC']
            l.append(evaluate_metric_with_gt(a_, metric, labels_))

        results[metric] = l

    # Save
    results_path = os.path.join(path_sample, f'evaluation_metrics_aucpr_{sample}_{filtering}.pickle')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    ##

    ############################## 
    # 2. What kNN graph (fixed k, varying metric) has the
    #  best association with gt clonal labels (kBET metric)?
    ############################## 

    # Here we go
    results = {}
    metrics = [
        'euclidean', 'sqeuclidean', 'cosine', 
        'correlation', 'jaccard', 'matching', 'ludwig2019'
    ]

    # Compute kNNs and corresponding idx
    k = 30
    labels = a.obs['GBC'].astype('str')

    for metric in metrics:
        if metric != 'ludwig2019':
            idx = kNN_graph(a.X, k=k, nn_kwargs={'metric':metric})[0]
        else:
            X = pair_d(a, metric='ludwig2019')
            idx = kNN_graph(X, k=k, from_distances=True)[0]

        _, _, acc_rate = kbet(idx, labels, alpha=0.05, only_score=False)
        median_entropy = NN_entropy(idx, labels)
        median_purity = NN_purity(idx, labels)
        results[metric] = {
            'kBET_rejection_rate' : 1-acc_rate, 
            'median_NN_entropy': median_entropy, 
            'median_NN_purity' : median_purity
        }

    ##

    # Save
    results_path = os.path.join(path_sample, f'evaluation_metrics_kNN_{sample}_{filtering}.pickle')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    ############################## 
    # 3. What is the relationship among all different metrics, 
    # considering their resulting fixed size kNN neighborhoods?
    ############################## 

    # Here we go
    results = {}
    metrics = [
        'euclidean', 'sqeuclidean', 'cosine', 
        'correlation', 'jaccard', 'matching', 'ludwig2019'
    ]

    # Compute kNNs
    k = 30
    for metric in metrics:
        if metric != 'ludwig2019':
            results[metric] = kNN_graph(a.X, k=k, nn_kwargs={'metric':metric})[0]
        else:
            X = pair_d(a, metric='ludwig2019')
            results[metric] = kNN_graph(X, k=k, from_distances=True)[0]

    # Evaluate median k-neighborhood overlaps
    n = len(metrics)
    J = np.zeros((n,n))

    for i, x in enumerate(metrics):
        for j, y in enumerate(metrics):
            J[i,j] = np.median(
                np.sum(results[x][:,1:] == results[y][:,1:],
                axis=1)
            )
    df = pd.DataFrame(J, index=metrics, columns=metrics)

    # Viz overlaps
    fig, ax = plt.subplots(figsize=(5,5))
    plot_heatmap(df, ax=ax, label='n', title=f'{sample}, {filtering}: shared NN (median)')
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            path_sample, f'kNN_overlaps_{sample}_{filtering}.png'
        ),
        dpi=300
    )

    ##

#####################################################

# Run
if __name__ == "__main__":
    main()


