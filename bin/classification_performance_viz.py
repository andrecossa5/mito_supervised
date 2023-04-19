"""
Visualization of clones and samples classification performances.
"""

# Code
import pickle
import re
import os
import sys
import gc
from itertools import chain
from Cellula.plotting._plotting import *
from Cellula.plotting._plotting_base import *
from Cellula.plotting._colors import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from MI_TO.preprocessing import *
from MI_TO.diagnostic_plots import sturges
from MI_TO.heatmaps_plots import *
from MI_TO.utils import *
from MI_TO.diagnostic_plots import *
matplotlib.use('macOSX')


##


# Set paths
path_main = sys.argv[1]
heatmap_sample = sys.argv[2]
fast = sys.argv[3]

# heatmap_sample = 'MDA'
# fast = 'fast'
# path_main = '/Users/IEO5505/Desktop/MI_TO/'

path_clones = path_main + '/results_and_plots/clones_classification/'
path_samples = path_main + '/results_and_plots/samples_classification/'
path_results = path_main + '/results_and_plots/classification_performance/'

# Read reports
clones = pd.read_excel(path_clones + 'report_classification_clones.xlsx', index_col=0)
samples = pd.read_excel(path_samples + 'report_classification_samples.xlsx', index_col=0)

# Re-format analysis
clones['analysis'] += '_' + clones['model']
samples['analysis'] += '_' + samples['model']

############## 


##


############## f1 by clone and feat_type
fig, ax = plt.subplots(figsize=(12, 5))

feat_type_colors = create_palette(clones, 'feature_type', 'Set1')
params = {   
            'showcaps' : True,
            'fliersize': 0,
            'boxprops' : {'edgecolor': 'black', 'linewidth': 0.3}, 
            'medianprops': {"color": "black", "linewidth": 1},
            'whiskerprops':{"color": "black", "linewidth": 1}
        }
box(clones, 'comparison', 'f1', c='#E9E7E7', ax=ax, kwargs=params)
strip(clones, 'comparison', 'f1', by='feature_type', c=feat_type_colors, s=2, ax=ax)
format_ax(clones, ax, title='f1-scores by clone and variant selection method', rotx=90, xsize=5)
create_handles(feat_type_colors.keys(), marker='o', colors=None, size=10, width=0.5)
handles = create_handles(feat_type_colors.keys(), colors=feat_type_colors.values())
fig.legend(handles, feat_type_colors.keys(), loc='upper right', 
    bbox_to_anchor=(0.9, 0.9), ncol=2, frameon=False, title='Feature selection'
)

v = 0.8
np.sum([ np.sum(clones.query('comparison == @x')['f1'] > v) > 0 for x in clones['comparison'].unique() ])

v = 0.5
np.sum([ np.median(clones.query('comparison == @x')['f1']) > v for x in clones['comparison'].unique() ])

ax.text(0.25, 0.8, f'-n clones with more than one classification above 0.5 f1: 6', transform=ax.transAxes)
ax.text(0.25, 0.75, f'-n clones with more than one classification above 0.8 f1: 6', transform=ax.transAxes)
ax.text(0.25, 0.7, f'-n clones with median f1 > 0.5: 1', transform=ax.transAxes)

# Save
fig.tight_layout()
fig.savefig(path_results + 'clones_f1.pdf')
##############


##


############## f1 by sample and feat_type
fig, ax = plt.subplots(figsize=(8, 6.8))

box(clones, 'sample', 'f1', ax=ax, s=0.5, c='#E9E7E7', params=params)
strip(clones, 'sample', 'f1', by='feature_type', c=feat_type_colors, s=3, ax=ax)
format_ax(clones, ax, title='Clones f1-scores by sample and variant selection method', ylabel='f1')
create_handles(feat_type_colors.keys(), marker='o', colors=None, size=10, width=0.75)
handles = create_handles(feat_type_colors.keys(), colors=feat_type_colors.values())
fig.subplots_adjust(right=0.75)
fig.legend(handles, feat_type_colors.keys(), loc='center', 
    bbox_to_anchor=(0.85, 0.6), ncol=1, frameon=False, title='Feature selection'
)
ax.text(1.05, 0.4, f'Mean MDA: {clones.loc[clones["sample"]=="AML"]["f1"].mean():.3f}', transform=ax.transAxes)
ax.text(1.05, 0.36, f'Mean AML: {clones.loc[clones["sample"]=="MDA"]["f1"].mean():.3f}', transform=ax.transAxes)
ax.text(1.05, 0.32, f'Mean PDX: {clones.loc[clones["sample"]=="PDX"]["f1"].mean():.3f}', transform=ax.transAxes)

# Save
fig.savefig(path_results + 'clones_f1_by_sample.pdf')
##############


##



############## f1 by model and feat_type
fig, ax = plt.subplots(figsize=(8, 6.5))

box(clones, 'model', 'f1', ax=ax, c='#E9E7E7', params=params)
strip(clones, 'model', 'f1', by='feature_type', c=feat_type_colors, s=3, ax=ax)
format_ax(clones, ax, title='Clones f1-scores by model and variant selection method', ylabel='f1')
create_handles(feat_type_colors.keys(), marker='o', colors=None, size=10, width=0.5)
handles = create_handles(feat_type_colors.keys(), colors=feat_type_colors.values())
fig.subplots_adjust(right=0.75)
fig.legend(handles, feat_type_colors.keys(), loc='center', 
    bbox_to_anchor=(0.85, 0.6), ncol=1, frameon=False, title='Feature selection'
)
ax.text(1.05, 0.39, f'Mean xgboost: {clones.loc[clones["model"]=="xgboost"]["f1"].mean():.3f}', transform=ax.transAxes)
ax.text(1.05, 0.36, f'Mean logit: {clones.loc[clones["model"]=="logit"]["f1"].mean():.3f}', transform=ax.transAxes)

# Save
fig.savefig(path_results + 'clones_f1_by_model.pdf')
##############


##


############## f1 by sample and feat_type
# Sizes
res = []
for x in os.listdir(path_main + '/data/CBC_GBC_cells/'):
    if x.endswith('csv'):
        d = pd.read_csv(path_main + f'/data/CBC_GBC_cells/{x}', index_col=0)
        res.append(d.assign(sample=x.split('_')[-1].split('.')[0]))
CBC_GBC = pd.concat(res, axis=0)
clones_sizes = CBC_GBC.groupby('GBC').size()

clones['GBC'] = clones['comparison'].map(lambda x: x.split('_')[0])

csizes = []
for x in clones['GBC']:
    csizes.append(clones_sizes[x])
clones['size'] = csizes

# Viz
fig, ax = plt.subplots(figsize=(6, 6))
scatter(clones, 'size', 'f1', c='#606060', s=3, ax=ax)
x = clones['size']
y = clones['f1']
fitted_coefs = np.polyfit(x, y, 1)
y_hat = np.poly1d(fitted_coefs)(x)
ax.plot(x, y_hat, linestyle='dotted', linewidth=2, color='r')
corr = np.corrcoef(x, y)[0,1]
ax.text(0.6, 0.9, f"Pearson's r: {corr:.2f}", transform=ax.transAxes)
format_ax(clones, ax, title='f1-clone size correlation', xlabel='Clone size', ylabel='f1')

# Save
fig.tight_layout()
fig.savefig(path_results + 'clones_size_f1_corr.pdf')
##############



##


############## 
# For each sample (3x) clones, what are the top 3 analyses (median f1 score across clones)? 
# Intersection among selected SNVs??

# Save top3 for easy quering
sample_names = clones['sample'].unique()
top_3 = {}
for sample in sample_names:
    top_3[sample] = clones.query('sample == @sample').groupby(['analysis']).agg(
        {'f1':np.median}).sort_values(
        'f1', ascending=False).index[:3].to_list()

with open(path_clones + 'top3.pkl', 'wb') as f:
    pickle.dump(top_3, f)

# Load top3 variants for each sample clones, and visualize their intersection (i.e., J.I.), by sample
top3_sample_variants = {}
for sample in sample_names:
        var_dict = {}
        top_3_sample = top_3[sample]
        for x in os.listdir(path_clones):
            if bool(re.search('|'.join(top_3_sample), x)):
                n = '_'.join(x.split('.')[0].split('_')[2:-2])
                df_ = pd.read_excel(path_clones + x, index_col=0)
                var_dict[n] = df_.index.unique().to_list()
        top3_sample_variants[sample] = var_dict

# Sample a
fig, axs = plt.subplots(1, 3, figsize=(12,5))

for k, sample in enumerate(top3_sample_variants):
    n_analysis = len(top3_sample_variants[sample].keys())
    JI = np.zeros((n_analysis, n_analysis))
    for i, l1 in enumerate(top3_sample_variants[sample]):
        for j, l2 in enumerate(top3_sample_variants[sample]):
            x = top3_sample_variants[sample][l1]
            y = top3_sample_variants[sample][l2]
            JI[i, j] = ji(x, y)
    JI = pd.DataFrame(data=JI, index=None, columns=top3_sample_variants[sample].keys())

    plot_heatmap(JI, palette='mako', ax=axs[k], title=sample, y_names=False,
        x_names_size=10, y_names_size=0, annot=True, annot_size=10, cb=True, label='JI variants'
    )

fig.tight_layout()
fig.savefig(path_results + 'overlap_selected_vars.pdf')
##############


##



############## f1 by clone, only top3 variants
heatmap_sample = 'MDA'
top_l = top_3[heatmap_sample]
df_ = clones.query('sample == @heatmap_sample and analysis in @top_l')
colors = {'MDA':'#DA5700', 'AML':'#0074DA', 'PDX':'#0F9221'}

fig, ax = plt.subplots(figsize=(6, 6))

box(df_, 'comparison', 'f1', c='#E9E7E7', ax=ax)
strip(df_, 'comparison', 'f1', s=6, c=colors[heatmap_sample], ax=ax)
clone_names = [ f'{x.split("_")[0][:5]}...' for x in df_['comparison'].unique() ] 
format_ax(ax, title=f'{heatmap_sample} clones f1-scores (only top analyses)', xticks=clone_names, 
    xlabel='Clones', ylabel='f1', xticks_size=5)
ax.set(ylim=(0,1.05))

perfs = clones.query('sample == @heatmap_sample').groupby(
    ['analysis']).agg({'f1':np.median}).sort_values(by='f1', ascending=False).values
ax.text(0.1, 0.86, 'Top analyses:', transform=ax.transAxes)
ax.text(0.1, 0.82, f'1. {top_l[0]} (median f1: {perfs[0][0]:.2f})', transform=ax.transAxes)
ax.text(0.1, 0.78, f'2. {top_l[1]} (median f1: {perfs[1][0]:.2f})', transform=ax.transAxes)
ax.text(0.1, 0.74, f'3. {top_l[2]} (median f1: {perfs[2][0]:.2f})', transform=ax.transAxes)

# Save
fig.tight_layout()

plt.show()
fig.savefig(path_results + f'{heatmap_sample}_clones_f1_only_top3.pdf')
# ##############


##
plt.show()


############## 
# For the sample top3 analysis on the clone task, what are the AF profiles of the variants selected?
# Which relatinship can we visualize among clone cells, using:
# 1) hclustering of cell x var AFM 
# 2) hclustering of a cell x cell similarity matrix?
path_data = path_main + 'data/'
path_distances = path_main + 'results_and_plots/distances/'

# Here we go
if not os.path.exists(path_results + 'top_3'):
    os.mkdir(path_results + 'top_3')
os.chdir(path_results + 'top_3')

if not os.path.exists(heatmap_sample):
    os.mkdir(heatmap_sample)
os.chdir(heatmap_sample)

# Read data and create colors
afm = read_one_sample(path_main, heatmap_sample)
clone_colors = create_palette(afm.obs, 'GBC', palette=sc.pl.palettes.default_20)
gc.collect()

# Fast or not?
L = [ '_'.join(x.split('_')[1:-1]) for x in top_3[heatmap_sample] ]
if fast == 'fast':
    to_run = [ L[0] ]
else:
    to_run = L

# For the analysis to_run
for analysis in to_run:
    print(analysis)

    a_ = analysis.split('_') 
    filtering = a_[0]
    min_cell_number = int(a_[1])
    min_cov_treshold = int(a_[2])

    # Filter cells and vars
    a_cells, a = filter_cells_and_vars(
        afm, 
        variants=top3_sample_variants[heatmap_sample][analysis],
        min_cell_number=min_cell_number,
        min_cov_treshold=min_cov_treshold,
    )
    gc.collect()

    # Control vars...
    print(analysis)
    print(a)
    assert all([ var in top3_sample_variants[heatmap_sample][analysis] for var in a.var_names ])

    # Get info!
    if not os.path.exists(path_results + f'top_3/{heatmap_sample}/{analysis}/'):

        os.mkdir(path_results + f'top_3/{heatmap_sample}/{analysis}/')
        os.chdir(path_results + f'top_3/{heatmap_sample}/{analysis}/')

        # 1-Viz selected variants properties
        fig, axs = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True) 
        
        colors = {'non-selected':'grey', 'selected':'red'} 
        to_plot = a_cells.copy()
        to_plot.X[np.isnan(to_plot.X)] = 0 

        # Vafs distribution
        for i, var in enumerate(a_cells.var_names):
            x = to_plot.X[:, i]
            x = np.sort(x)
            if var in a.var_names:
                axs[0].plot(x, '--', color=colors['selected'], linewidth=0.5)
            else:
                axs[0].plot(x, '--', color=colors['non-selected'], linewidth=0.2)
        format_ax(pd.DataFrame(x), ax=axs[0], title='Ranked AFs', xlabel='Cell rank', ylabel='AF')

        # Vafs summary stats
        df_ = summary_stats_vars(to_plot, variants=None).drop('median_coverage', axis=1).reset_index(
            ).rename(columns={'index' : 'variant'}).assign(
            is_selected=lambda x: np.where(x['variant'].isin(a.var_names), 'selected', 'non-selected')).melt(
            id_vars=['variant', 'is_selected'], var_name='summary_stat')

        box(df_, 'summary_stat', 'value', by='is_selected', c=colors, ax=axs[1], params=params)
        format_ax(df_, ax=axs[1], title='Summary statistics', 
            xticks=df_['summary_stat'].unique(), xlabel='', ylabel='Value'
        )
        handles = create_handles(colors.keys(), marker='o', colors=colors.values(), size=10, width=0.5)
        axs[1].legend(handles, colors.keys(), title='Selection', loc='center left', 
            bbox_to_anchor=(1, 0.5), ncol=1, frameon=False
        )
        fig.suptitle(f'{heatmap_sample}: analysis {analysis}')

        # Save
        fig.savefig(f'{analysis}_variants.pdf')

        # 2-Viz cell x var and cell x cell heatmaps
        with PdfPages(f'{heatmap_sample}_{analysis}_heatmaps.pdf') as pdf:

            a = nans_as_zeros(a)
            cell_anno_clones = [ clone_colors[clone] for clone in a.obs['GBC'] ]

            # Viz 
            if a.var_names.size < 100:
                size = 3
            else:
                size = 1
            g = cells_vars_heatmap(a, cell_anno=cell_anno_clones, 
                anno_colors={ k:v for k,v in clone_colors.items() if k in a.obs['GBC'].unique() }, 
                heat_label='AF', legend_label='Clone', figsize=(11, 8), title=f'{heatmap_sample}: {analysis}',
                xticks_size=size
            )
            pdf.savefig() 
            
            # Prep d for savings
            analysis_d = {}
            analysis_d['cells'] = a.obs_names.to_list()
            analysis_d['vars'] = a.var_names.to_list()
            analysis_d['dendrogram'] = g.dendrogram_row.dendrogram
            analysis_d['linkage'] = g.dendrogram_row.linkage
            
            with open('cell_x_var_hclust.pickle', 'wb') as f:
                pickle.dump(analysis_d, f)

            # 3-Viz all cell x cell similarity matrices obtained from the filtered AFM one.
            for x in os.listdir(path_distances):
                if bool(re.search(f'{heatmap_sample}_{analysis}_', x)):

                    print(x)
                    a_ = x.split('_')[:-1]
                    metric = a_[-1]
                    with_nans = 'w/i nans' if a_[-2] == 'yes' else 'w/o nans'
                    D = sc.read(path_distances + x)
                    gc.collect()

                    if a.shape[0] == D.shape[0]:
                        print(a)
                        print(D)
                        assert (a.obs_names == D.obs_names).all()
                        D.obs['GBC'] = a.obs['GBC']

                        # Draw clustered similarity matrix heatmap 
                        heat_title = f'{heatmap_sample} clones: {filtering}_{min_cell_number}_{min_cov_treshold}, {metric} {with_nans}'
                        g = cell_cell_dists_heatmap(D, 
                            cell_anno=cell_anno_clones, 
                            anno_colors={ k:v for k,v in clone_colors.items() if k in a.obs['GBC'].unique() },
                            heat_label='Similarity', legend_label='Clone', figsize=(11, 6.5), 
                            title=heat_title
                        )
                        pdf.savefig()

                        analysis_d = {}
                        analysis_d['dendrogram'] = g.dendrogram_row.dendrogram
                        analysis_d['linkage'] = g.dendrogram_row.linkage

                        with open(f'similarity_{"_".join(a_)}_hclust.pickle', 'wb') as f:
                            pickle.dump(analysis_d, f)

                    else:
                        print(f'{x} not added...')

            plt.close()

    else:
        print(f'Analysis {analysis} hclusts have been already computed...')
#############


##


############# 
# For each sample top3 analysis on the clone task, what is the number of selected variants?
d = {}
for sample in top3_sample_variants:
    n_vars = {}
    for analysis in top3_sample_variants[sample]:
        n_vars[analysis] = len(top3_sample_variants[sample][analysis])
    d[sample] = n_vars
df_ = pd.DataFrame(d).reset_index().rename(columns={'index':'analysis'}).melt(
    id_vars='analysis', var_name='sample', value_name='n_vars').dropna()

# Viz 
colors = {'MDA':'#DA5700', 'AML':'#0074DA', 'PDX':'#0F9221'}
fig, ax = plt.subplots(figsize=(6,7), constrained_layout=True)
bar(df_, 'n_vars', x=None, by='sample', c=colors, ax=ax, s=0.75, annot_size=10)
format_ax(df_, ax=ax, xticks=df_['analysis'], rotx=90, 
    ylabel='n variants', title='n variants selected by the top 3 analyses'
)
handles = create_handles(colors.keys(), marker='o', colors=colors.values(), size=10, width=0.5)
ax.legend(handles, colors.keys(), title='Sample', loc='center', 
    bbox_to_anchor=(1.1, 0.5), ncol=1, frameon=False
)

# Save
fig.savefig(path_results + 'n_top3_selected_variants.pdf')
################


##


# ############## 
# For each sample (3x) clones, what are the clones that are consistently predictable in the top analyses? 
# What are their features? 
top_clones_d = {}
for sample in clones['sample'].unique():
    top_analyses = top_3[sample]
    top_clones = clones.query('sample == @sample and analysis in @top_analyses').groupby(['comparison']).agg(
        {'f1':np.median}).sort_values(
        'f1', ascending=False).query('f1 > 0.5').index.to_list()
    top_clones_stats = clones.query(
        'sample == @sample and analysis in @top_analyses and comparison in @top_clones'
        )
    top_clones_d[sample] = {'clones' : top_clones, 'stats' : top_clones_stats}

print(f'Top clones: {top_clones_d}')

# Here we go...
if len(top_clones_d[heatmap_sample]['clones']) > 0:

    best_for_each_top_clone_df = top_clones_d[heatmap_sample]['stats'].sort_values(
        'f1', ascending=False).groupby('comparison').head(1).loc[
            :, ['feature_type', 'min_cell_number', 'min_cov_treshold', 'comparison', 'model']
        ].assign(
            clone=lambda x: x['comparison'].map(lambda y: y.split('_')[0])
        ).set_index('clone').drop('comparison', axis=1)

    # For each top classified clone in that sample, and its top analysis...
    for i in range(best_for_each_top_clone_df.shape[0]):
        # Get top clone id, and its top classification analysis options
        topper = best_for_each_top_clone_df.index[i]
        filtering = best_for_each_top_clone_df['feature_type'][i]
        min_cell_number = best_for_each_top_clone_df['min_cell_number'][i]
        min_cov_treshold = best_for_each_top_clone_df['min_cov_treshold'][i]
        model = best_for_each_top_clone_df['model'][i]

        # Viz
        print(topper)
        fig = viz_clone_variants(
            afm, topper, 
            sample=heatmap_sample, path=path_clones, 
            filtering=filtering, min_cell_number=min_cell_number, 
            min_cov_treshold=min_cov_treshold, model=model, 
            figsize=(10,10)
        )
        
        fig.savefig(path_results + f'top_3/{heatmap_sample}/{topper}_features.pdf')
################


##


################ Method rankings, so far

c = { 0 : '#097CAE', 50 : '#09AE50'}
for sample in ['AML', 'MDA', 'PDX']:
    
    for min_cell_number in [0, 50]:

        fig, ax = plt.subplots(figsize=(8, 5))

        query = 'sample == @sample and min_cell_number == 50'
        idx = clones.query(query).groupby(
            'feature_type').median().sort_values(by='f1', ascending=False).index

        sns.boxplot(clones.query(query), x='feature_type', y='f1',
            ax=ax, saturation=0.55, order=idx, color=c[min_cell_number])
        format_ax(ax, xlabel='', title=f'{sample}: {min_cell_number} min_cell_number')
        
        fig.savefig(path_results + f'{sample}_{min_cell_number}.pdf')
        

