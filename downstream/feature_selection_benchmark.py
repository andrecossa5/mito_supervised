"""
Feature selection benchmark.
"""

import os
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.plotting_base import *
matplotlib.use('macOSX')


##


# Set paths
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data') 
path_results = os.path.join(path_main, 'results', 'var_selection')


##


# Load variants
with open(os.path.join(path_results, 'GT_enriched_variants.pickle'), 'rb') as f:
    d_enriched = pickle.load(f)
with open(os.path.join(path_results, 'GT_stringent_variants.pickle'), 'rb') as f:
    d_stringent = pickle.load(f)
with open(os.path.join(path_results, 'variants.pickle'), 'rb') as f:
    d = pickle.load(f)


##


# Gather metrics
L = []
for k in d:

    sample, filtering_key = k 
    variant_set = d[k]

    L.append({ 'value': np.log10(len(variant_set)), 'GT':None, 'sample':sample, 'filtering_key': filtering_key, 'metric':'log10(n vars)' }) 
    # L.append({ 'value': ji(variant_set, d_stringent[(sample, 'stringent')]), 'GT':'stringent', 'sample':sample, 'filtering_key': filtering_key, 'metric':'JI' }) 
    # L.append({ 'value': ji(variant_set, d_enriched[(sample, 'enriched')]),  'GT':'enriched','sample':sample, 'filtering_key': filtering_key, 'metric':'JI' }) 

    tpr = lambda x, y: len(set(x) & set(y)) / len(x)
    L.append({ 'value': tpr(d_stringent[(sample, 'stringent')], variant_set), 'GT':'stringent', 'sample':sample, 'filtering_key': filtering_key, 'metric':'TPR' }) 
    L.append({ 'value': tpr(d_enriched[(sample, 'enriched')], variant_set), 'GT':'enriched', 'sample':sample, 'filtering_key': filtering_key, 'metric':'TPR' }) 

    fpr = lambda x, y: len(set(x) - set(y)) / len(x)
    L.append({ 'value': fpr(variant_set, d_stringent[(sample, 'stringent')]), 'GT':'stringent', 'sample':sample, 'filtering_key': filtering_key, 'metric':'FPR' }) 
    L.append({ 'value': fpr(variant_set, d_enriched[(sample, 'enriched')]), 'GT':'enriched', 'sample':sample, 'filtering_key': filtering_key, 'metric':'FPR' }) 

    fnr = lambda x, y: len(set(x) - set(y)) / len(x)
    L.append({ 'value': fpr(d_stringent[(sample, 'stringent')], variant_set), 'GT':'stringent', 'sample':sample, 'filtering_key': filtering_key, 'metric':'FNR' }) 
    L.append({ 'value': fpr(d_enriched[(sample, 'enriched')], variant_set), 'GT':'enriched', 'sample':sample, 'filtering_key': filtering_key, 'metric':'FNR' }) 

# Df and colors
df = pd.DataFrame(L)
colors = create_palette(df, 'filtering_key', ten_godisnot)


##


# Fig
fig = plt.figure(figsize=(10,5))

for i, x in enumerate(df['metric'].unique()):

    df_ = df.query('metric==@x')
    order = df_.groupby('filtering_key')['value'].median().sort_values(ascending=False).index
    ax = plt.subplot(2,2,i+1)
    box(df_, 'filtering_key', 'value', ax=ax, c=colors, order=order, a=.1)
    strip(df_, 'filtering_key', 'value', c=colors, ax=ax, order=order, s=4)
    format_ax(ax, ylabel=x, rotx=90)
    ax.spines[['right', 'top']].set_visible(False)

fig.tight_layout()
fig.savefig(
    os.path.join(path_results, f'filtering_methods_and_GT.png'), 
    dpi=500
)


##