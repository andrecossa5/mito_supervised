"""
Investigation of MT-SNVs metrics spaces.
1. What is the best performing metric (i.e., the one that achieve better AUCPR on gt clonal labels)?
2. What kNN graph (fixed k, varying metric) has the best association with gt clonal labels?
3. What is the relationship among all differnt metrics, considering the resulting kNN graphs?
"""


# Code
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


# Args
path_main = sys.argv[1]
sample = sys.argv[2]
filtering = sys.argv[3]


##


# Set paths
path_data = os.path.join(path_main, 'data')
path_viz = os.path.join(path_main, 'results', 'supervised_clones', 'visualization', 'distances_evaluation')
path_tmp = os.path.join(path_main, 'results', 'supervised_clones', 'downstream_files')


##


def main():

    # Data
    make_folder(path_tmp, sample, overwrite=False)
    path_sample = os.path.join(path_tmp, sample)
    afm = read_one_sample(path_data, sample, with_GBC=True)

    # Filter AFM
    file = f'{sample}_filtered_subsets.pickle'
    with open(os.path.join(path_tmp, 'variant_subsets_and_GT', file), 'rb') as f:
        d_variants = pickle.load(f)
    _, a = filter_cells_and_vars(
        afm, sample=sample, variants=d_variants[filtering],
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
    with open(os.path.join(path_sample, f'evaluation_metrics_aucpr_{sample}_{filtering}.pickle'), 'wb') as f:
        pickle.dump(results, f)

    # Metric order
    df_ = pd.DataFrame(results).melt(var_name='metric', value_name='AUCPR')
    order = (
        df_.groupby('metric')
        .agg('median')['AUCPR']
        .sort_values(ascending=False)
        .index
    )

    # Viz
    fig, ax = plt.subplots(figsize=(7,5))
    box(df_, x='metric', y='AUCPR', c='white', ax=ax, order=order)
    strip(df_, x='metric', y='AUCPR', c='black', s=5, ax=ax, order=order)
    format_ax(
        ax,
        title=f'AUCPR all positive cell pairs (n samples={n_samples})', 
        ylabel='AUCPR',
        reduced_spines=True
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(path_viz, f'evaluation_metrics_aucpr_{sample}_{filtering}.png'),
        dpi=500
    )

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

        mean_ksqared, mean_p, acc_rate = kbet(idx, labels, alpha=0.05, only_score=False)
        median_entropy = NN_entropy(idx, labels)
        median_purity = NN_purity(idx, labels)
        results[metric] = {
            'kBET_rejection_rate' : 1-acc_rate, 
            'median_NN_entropy': median_entropy, 
            'median_NN_purity' : median_purity
        }

    ##

    # Save
    with open(os.path.join(path_sample, f'evaluation_metrics_kNN_{sample}_{filtering}.pickle'), 'wb') as f:
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
            path_viz, f'kNN_overlaps_{sample}_{filtering}.png'
        ),
        dpi=500
    )

    ##

#####################################################

# Run
if __name__ == "__main__":
    main()


