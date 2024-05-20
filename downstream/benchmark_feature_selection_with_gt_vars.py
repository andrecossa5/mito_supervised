"""
UMAP samples from GT variants.
"""

import os
from mito_utils.utils import *
from plotting_utils._plotting_base import *
matplotlib.use('macOSX')


##


# Set paths
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data') 
path_results = os.path.join(path_main, 'results', 'supervised_clones')


##


# Load GT variants and clone colors
with open(os.path.join(path_results, 'variants.pickle'), 'rb') as f:
    variants = pickle.load(f)


##
    

# Convert in a sample : dict dictionary
d = {}
samples = np.unique([ x[0] for x in variants ])
for sample in samples:
    sets = {}
    for k in variants:
        if k[0] == sample:
            sets[k[1]] = variants[k]
    d[sample] = sets

# Gather metrics
L = []
for k in d:
    d_ = d[k]
    filtered_subsets_keys = [ x for x in d_ if x != 'GT' ]
    jis = { x : ji(d_['GT'], d_[x]) for x in filtered_subsets_keys }
    jis = { **jis, **{ 'sample' : k, 'metric' : 'JI' } }
    L.append(jis)
    tps = { x : len(set(d_['GT']) & set(d_[x])) / len(d_['GT']) for x in filtered_subsets_keys }
    tps = { **tps, **{ 'sample' : k, 'metric' : 'TPR' } }
    L.append(tps)
    fps = { x : len(set(d_[x]) - set(d_['GT'])) / len(d_[x]) for x in filtered_subsets_keys }
    fps = { **fps, **{ 'sample' : k, 'metric' : 'FPR' } }
    L.append(fps)
    fns = { x : len(set(d_['GT']) - set(d_[x])) / len(d_['GT']) for x in filtered_subsets_keys }
    fns = { **fns, **{ 'sample' : k, 'metric' : 'FNR' } }
    L.append(fns)

df = (
    pd.DataFrame(L)
    .melt(value_name='value', var_name='method', id_vars=['sample', 'metric'])
)


##
    

# Viz
colors = {
    'MQuad': '#ff7f0e', 'pegasus': '#279e68', 
    'ludwig2019': '#d62728', 'miller2022': '#aa40fc'
}

# Fig
fig = plt.figure(figsize=(8,5))
order = ( 
    df.query('metric=="TPR"')
    .groupby('method').mean()
    .sort_values(by='value', ascending=False).index
)
for i, x in enumerate(df['metric'].unique()):

    df_ = df.query('metric == @x')
    ax = plt.subplot(2,2,i+1)
    box(df_, 'method', 'value', ax=ax, c=colors, order=order, alpha=.5)
    strip(df_, 'method', 'value', c=colors, ax=ax, order=order, s=7)
    format_ax(ax, ylabel=x)
    ax.spines[['right', 'top']].set_visible(False)

fig.tight_layout()
fig.savefig(
    os.path.join(path_results, f'filtering_methods_and_GT.pdf'), 
    dpi=500
)


##