"""
Script to visualize GT variants.
"""

import os
import warnings
from mito_utils.preprocessing import *
from mito_utils.clustering import *
from mito_utils.dimred import *
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.plotting_base import *
from mito_utils.diagnostic_plots import *
from mito_utils.embeddings_plots import *
from mito_utils.kNN import kNN_graph
from mito_utils.metrics import NN_purity
from mito_utils.distances import pair_d
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('macOSX')


##


# Args
path_main = '/Users/IEO5505/Desktop/mito_bench'


##


# Set paths
path_data = os.path.join(path_main, 'data') 
path_output = os.path.join(path_main, 'results', 'supervised_clones', 'output')
path_viz = os.path.join(
    path_main, 'results', 'supervised_clones', 'visualization', 'variant_subsets_and_GT'
)
path_tmp = os.path.join(
    path_main, 'results', 'supervised_clones', 'downstream_files', 'variant_subsets_and_GT'
)


##



################################# Filtering MT-SNVs vs GT

samples = ['MDA_clones'] # , 'MDA_PT']

##

def extract_vars(path_tmp, samples):

    D = {}
    for sample in samples:

        d_vars = {}
        for x in os.listdir(path_tmp):
            if x == f'{sample}_GT_variants.pickle':
                with open(os.path.join(path_tmp, x), 'rb') as f:
                    d = pickle.load(f)
                GT_vars = d['(0.75,0.25)']
        d_vars['GT'] = GT_vars

        for x in os.listdir(path_tmp):
            if x == f'{sample}_filtered_subsets.pickle':
                with open(os.path.join(path_tmp, x), 'rb') as f:
                    d = pickle.load(f)

        D[sample] = { **d_vars, **d }

    return D


##


def gather_metrics(d):

    L = []

    for k in d:

        d_ = d[k]
        filtered_subsets_keys = [ x for x in d_ if x != 'GT' ]

        jis = { x : ji(d_['GT'], d_[x]) for x in filtered_subsets_keys }
        jis = { **jis, **{ 'sample' : k, 'metric' : 'JI' } }
        L.append(jis)

        tps = { x : len(set(d_['GT']) & set(d_[x])) / len(d_['GT'])  for x in filtered_subsets_keys }
        tps = { **tps, **{ 'sample' : k, 'metric' : 'TPR' } }
        L.append(tps)

        fps = { x : len(set(d_[x]) - set(d_['GT'])) / len(d_[x])  for x in filtered_subsets_keys }
        fps = { **fps, **{ 'sample' : k, 'metric' : 'FPR' } }
        L.append(fps)

        fns = { x : len(set(d_['GT']) - set(d_[x])) / len(d_['GT']) for x in filtered_subsets_keys }
        fns = { **fns, **{ 'sample' : k, 'metric' : 'FNR' } }
        L.append(fns)


    df = (
        pd.DataFrame(L)
        .melt(value_name='value', var_name='method', id_vars=['sample', 'metric'])
    )

    return df


##


# Get metrics
d = extract_vars(path_tmp, samples)
df = gather_metrics(d)

# Viz
fig = plt.figure(figsize=(8,5), constrained_layout=True)

for i, x in enumerate(df['metric'].unique()):

    df_ = df.query('metric == @x')
    order = ( 
        df_.groupby('method').mean()
        .sort_values(by='value', ascending=False).index
    )

    ax = plt.subplot(2,2,i+1)
    box(df_, 'method', 'value', c='white', ax=ax, order=order)
    strip(df_, 'method', 'value', c='k', ax=ax, order=order)
    format_ax(ax, reduced_spines=True, ylabel=x)

fig.suptitle('MT-SNVs filtering method sensitivity')
fig.savefig(
    os.path.join(path_viz, f'filtering_methods_and_GT.png'), 
    dpi=300
)


##


# Improvement with MQuad-opt
sample = 'MDA_clones'
afm = read_one_sample(path_data, sample, with_GBC=True)
a_cells, a = filter_cells_and_vars(afm, filtering='MQuad_optimized')
ji(d[sample]['MQuad'], d[sample]['GT'])
ji(d[sample]['MQuad'], a.var_names)
d[sample]['GT'].isin(a.var_names).sum()
d[sample]['GT'].isin(d[sample]['MQuad']).sum()


##


################################# Examples GT

# Load data
sample = 'MDA_clones'

afm = read_one_sample(path_data, sample, with_GBC=True)
with open(os.path.join(path_tmp, f'{sample}_GT_variants.pickle'), 'rb') as f:
    d = pickle.load(f)

# Filter and umap
a_cells, a = filter_cells_and_vars(afm, variants=d['(0.75,0.25)'])
a = nans_as_zeros(a)
X_umap, name_dims = reduce_dimensions(a, 'UMAP', metric='cosine', n_comps=2)
df_ = (
    pd.DataFrame(X_umap, columns=name_dims, index=a.obs_names)
    .join(pd.DataFrame(a.X, index=a.obs_names, columns=a.var_names))
    .join(a.obs)
)


##


# Viz clones
fig, ax = plt.subplots(figsize=(4.5,4.5))

draw_embeddings(
    df_, cat='GBC', ax=ax, 
    title=f'{sample}: lentiviral clones',
    axes_kwargs={'legend':False}, 
    s=30
)
ax.axis('off')
fig.tight_layout()
fig.savefig(
    os.path.join(path_viz, f'GBC_GT_{sample}_umap.png'),
    dpi=300
)


##

# Variants spatial distribution
fig, axs = plt.subplots(2,1,figsize=(6,4.5))

# a_base = filter_baseline(a_cells)

variants = df_.columns[2:-2].to_series().reset_index(drop=True)
# variants = a_base.var_names.to_series().reset_index(drop=True)

X = df_[variants].values
# X = a_cells.X
x = variants.map(lambda x: int(x.split('_')[0])).sort_values()

axs[0].plot(x, np.nanmean(X[:,x.index], axis=0), 'ko')
axs[0].spines[['right', 'left', 'top']].set_visible(False)
axs[0].set(ylabel='AF')
axs[0].grid(axis='x')

axs[1].plot(x, np.sum(X[:,x.index]>0, axis=0)/X.shape[0], 'ko')
axs[1].spines[['right', 'left', 'top']].set_visible(False)
axs[1].set(ylabel='% +cells', xlabel='Genomic coordinate (bp)')
axs[1].grid(axis='x')

fig.suptitle(f'MT-SNVs (n={x.size}) distribution across MT-genome')
fig.tight_layout()
fig.savefig(
    os.path.join(path_viz, f'GT_{sample}_distribution.png'),
    dpi=300
)


##


# Viz
df_.groupby('GBC').size().sort_values()
rank_clone_variants(a, group='CCCGCTGCGGCTGCTCCG')


# Viz vars
var = '2659_C>T'
clone = 'CCCGCTGCGGCTGCTCCG'

fig, axs = plt.subplots(1,3, figsize=(13.5,4.5))

draw_embeddings(
    df_, cont=var, ax=axs[0], 
    title=f'{clone}: {var}',
    cbar_kwargs={'vmin':.01, 'vmax':.1}
)
plot_exclusive_variant(a_cells, var, ax=axs[1])
df_clone = (
    df_.loc[:,[var, 'GBC']]
    .assign(cat=lambda x: np.where(x['GBC'] == clone, clone, 'other'))
)
box(df_clone, x='cat', y=var, c='white', ax=axs[2])
strip(df_clone, x='cat', y=var, c='k', s=2.5, ax=axs[2])
format_ax(axs[2], ylabel='AF', title=f'{var} AF', reduced_spines=True)

fig.tight_layout()
plt.show()



fig.savefig(
    os.path.join(path_viz, f'{sample}_{var}.png'),
    dpi=300
)


##


# Viz one distribution
fig, ax = plt.subplots(figsize=(5,4.5))
hist(df_, var, sturges(df_[var]), c='k', ax=ax)
format_ax(ax, title=f'{var} AF', reduced_spines=True, xlabel='AF', ylabel='ncells')
fig.savefig(
    os.path.join(path_viz, f'{sample}_{var}_hist.png'), 
    dpi=300
)


##


# Adding dirty variants, removing informative ones

# Load data
sample = 'MDA_PT'

afm = read_one_sample(path_data, sample, with_GBC=True)
with open(os.path.join(path_tmp, f'{sample}_GT_variants.pickle'), 'rb') as f:
    d = pickle.load(f)

# Filter and umap
a_cells, a = filter_cells_and_vars(afm, variants=d['(0.75,0.25)'])
a = nans_as_zeros(a)
X_umap, name_dims = reduce_dimensions(a, 'UMAP', metric='cosine', n_comps=2)
df_ = (
    pd.DataFrame(X_umap, columns=name_dims, index=a.obs_names)
    .join(pd.DataFrame(a.X, index=a.obs_names, columns=a.var_names))
    .join(a.obs)
)

# GT
gt_vars = df_.columns[2:-2]

# All the others
a_base = filter_baseline(a_cells)
top_vmr = (
    summary_stats_vars(a_base)
    .query('fr_positives>.1')['VMR_rank']
    .sort_values().index[:200]
)
top_vmr = top_vmr[~top_vmr.isin(gt_vars)]


##


# Viz clones
fig, axs = plt.subplots(1,4,figsize=(12,3))


# for i, x in enumerate([0, 10, 25, 50]):
for i, x in enumerate([0, 2, 4, 6]):
    # a_ = a[:, gt_vars.to_list()+top_vmr[:x].to_list()]
    a_ = a[:, a.var_names[~a.var_names.isin(np.random.choice(a.var_names, x, replace=False))]]
    print(a_.shape)
    a_ = nans_as_zeros(a_)
    X_umap, name_dims = reduce_dimensions(a_, 'UMAP', metric='cosine', n_comps=2)
    idx = kNN_graph(a_.X, k=15, nn_kwargs={'metric':'cosine'})[0]
    purity = NN_purity(idx, a_.obs['GBC'])
    df_ = (
        pd.DataFrame(X_umap, columns=name_dims, index=a_.obs_names)
        .join(pd.DataFrame(a_.X, index=a_.obs_names, columns=a_.var_names))
        .join(a_.obs)
    )

    draw_embeddings(
        df_, cat='GBC', ax=axs[i], 
        # title=f'n added:{x}; kNN purity: {purity:.2f}',
        title=f'n removed:{x}; kNN purity: {purity:.2f}',
        axes_kwargs={'legend':False}
    )

fig.subplots_adjust(top=.8, wspace=.3)
fig.suptitle(f'Perturbation to {sample} "highly-specific" MT-SNVs space')
fig.savefig(
    os.path.join(path_viz, f'{sample}_gt_space_perturbation.png'), 
    dpi=300
)


##


# Cluster muts 



a_base = nans_as_zeros(a_base)

a_filtered = filter_cells_and_vars(afm, filtering='MQuad_optimized', path_=os.getcwd())




D = pair_d(a_filtered.X.T, metric='correlation')
D = pd.DataFrame(D, index=a_filtered.var_names, columns=a_filtered.var_names)


order = fast_hclust_distance(D.values)


fig, ax = plt.subplots(figsize=(5,5))
plot_heatmap(D.iloc[order, order], ax=ax, x_names=False, y_names=False, palette='magma_r')
fig.tight_layout()

plt.show()

a_filtered.X


path_ = '/Users/IEO5505/Desktop/mito_bench/results/supervised_clones/downstream_files/variant_subsets_and_GT'

os.listdir(path_)
with open(os.path.join(path_, 'MDA_PT_filtered_subsets.pickle'), 'rb') as f:
    d = pickle.load(f)

d['MQuad']



df = pd.read_csv(os.path.join(os.getcwd(), 'MQuad_stats.csv'), index_col=0)
df = df.join(summary_stats_vars(a_base))
df = df.sort_values('fr_positives')


fig, ax = plt.subplots(figsize=(4.5,4.5))

ax.plot(df['fr_positives'], df['deltaBIC'], 'ko')
ax.set_ylim((-50,1500))
ax.set_xlim((-.01,.5))

plt.show()

df = df.dropna()

df_ = df.query('deltaBIC<1000')

df

np.corrcoef(df['VMR'], df['deltaBIC'])