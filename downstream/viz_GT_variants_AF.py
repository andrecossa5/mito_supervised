"""
AF specturm of MT-SNVs from different filtering methods.
"""

import os
from mito_utils.preprocessing import *
from mito_utils.dimred import *
from mito_utils.utils import *
from mito_utils.plotting_base import *
matplotlib.use('macOSX')


##


# Args
path_main = '/Users/IEO5505/Desktop/mito_bench'

##


# Set paths
path_data = os.path.join(path_main, 'data') 
path_results = os.path.join(path_main, 'results', 'supervised_clones')


##


# Load GT variants and clone colors
with open(os.path.join(path_results, 'variants.pickle'), 'rb') as f:
    d = pickle.load(f)


##


# For each sample 
sample = 'AML_clones'
afm = read_one_sample(path_data, sample, with_GBC=True)
a_cells, a = filter_cells_and_vars(afm, filtering='pegasus', min_cell_number=10)
a = nans_as_zeros(a)

## Colors
methods = ['GT', 'MQuad', 'pegasus', 'ludwig2019', 'miller2022']
# colors = { method:c for method, c in zip(methods, sc.pl.palettes.default_20) }
colors = {
    'GT': '#1f77b4', 'MQuad': '#ff7f0e', 
    'pegasus': '#279e68', 'ludwig2019': '#d62728', 'miller2022': '#aa40fc'
}
##

fig = plt.figure(figsize=(3.5*len(methods), 4))
for i,method in enumerate(methods):
    ax = fig.add_subplot(1,len(methods),i+1)
    variants = d[(sample, method)]
    a_ = a_cells[:,variants]
    mean_af = np.nanmean(a_.X, axis=1).mean()
    perc_positive = np.mean(np.sum(a_.X>0, axis=1) / a_.X.shape[0])
    for i in range(a_.shape[1]):
        x = a_[:, i].X.flatten()
        x = np.sort(x)
        ax.plot(x, '-', color=colors[method], linewidth=1.7)
    format_ax(
        ax=ax, title=f'{method} \n n: {a_.shape[1]}, mean AF: {mean_af:.2f}', 
        xlabel='Cell rank', ylabel='AF', title_size=12
    )

# fig.subplots_adjust
fig.suptitle(f'{sample} variants selection')
fig.subplots_adjust(left=.05, right=.95, top=.8, bottom=.15, wspace=.27)
fig.savefig(
    os.path.join(path_results, f'{sample}_AF_like_variants_by_method.pdf'),
    dpi=500
)


##