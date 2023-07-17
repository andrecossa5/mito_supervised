"""
Investigation of MT-SNVs metrics spaces. Only three, decently recovered clones, with diffent prevalences.
"""

# Code
import os
import pickle
from mito_utils.preprocessing import *
from mito_utils.clustering import *
from mito_utils.utils import *
from mito_utils.distances import *
from mito_utils.kNN import *
from mito_utils.metrics import *
from mito_utils.plotting_base import *
import matplotlib
matplotlib.use('macOSX')


##


# Args
sample = 'MDA_PT'
filtering = 'MQuad'
ncov = int('100')

# Set paths
path_main = '/Users/IEO5505/Desktop/mito_bench/'
path_data = os.path.join(path_main, 'data')
path_output = os.path.join(path_main, 'results', 'supervised_clones', 'output')
path_report = os.path.join(path_main, 'results', 'supervised_clones', 'reports')
path_viz = os.path.join(path_main, 'results', 'supervised_clones', 'visualization', 'distances_evaluation')
path_tmp = os.path.join(path_main, 'results', 'supervised_clones', 'downstream_files')


##


# Load data
make_folder(path_tmp, 'three_clones_MDA_PT', overwrite=True)
path_sample = os.path.join(path_tmp, sample)
afm = read_one_sample(path_data, sample, with_GBC=True)

# Vars
with open(os.path.join(path_tmp, 'MDA_PT_variants.pickle'), 'rb') as f:
    variants = pickle.load(f)
a_cells, a = filter_cells_and_vars(
    afm, sample=sample, variants=variants[filtering],
    min_cell_number=10, min_cov_treshold=50, nproc=4
)
a = nans_as_zeros(a)
labels = a.obs['GBC']

# Choose three clones 

# Load clones performance
clones = pd.read_csv(os.path.join(path_report, 'report_f1.csv'), index_col=0)

# (
#     clones.query('sample == "MDA_PT"')
#     .groupby('comparison')
#     .apply(lambda x: x.sort_values('AUCPR', ascending=False).head(3).median())
#     .sort_values('AUCPR', ascending=False)
#     .loc[:, ['AUCPR', 'clone_prevalence']]
# )
three_clones = [ 'CGGGAGCAGGACAGCGAC', 'CCACGTGCCCAGCTGCAC', 'AAGCCACGTCGAGCAATT' ]

# Subset matrix
cells = a.obs.loc[a.obs['GBC'].isin(three_clones)].index
a = a[cells, :]
a.uns['per_position_coverage'] = a.uns['per_position_coverage'].loc[cells, :] 
a.uns['per_position_quality'] = a.uns['per_position_quality'].loc[cells, :] 

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
        l.append(evaluate_metric_with_gt(a_, metric, labels_, ncov=ncov))
    results[metric] = l

# Viz
df_ = pd.DataFrame(results).melt(var_name='metric', value_name='AUCPR')
order = (
    df_.groupby('metric')
    .agg('median')['AUCPR']
    .sort_values(ascending=False)
    .index
)

# Viz
fig, ax = plt.subplots(figsize=(7,5))
box(df_, x='metric', y='AUCPR', c='lightgrey', ax=ax, order=order)
strip(df_, x='metric', y='AUCPR', c='black', ax=ax, order=order)
format_ax(ax, title=f'AUCPR all positive cell pairs (n samples={n_samples}, three clones PT)', ylabel='AUCPR')
fig.tight_layout()
ax.spines[['right', 'top']].set_visible(False)
fig.savefig(
    os.path.join(path_viz, f'evaluation_metrics_aucpr_three_clones_MQuad.png'),
    dpi=500
)


##