#!/usr/bin/python

# Top models final assessment

########################################################################

import argparse
import os
import re
import pickle
import matplotlib

from Cellula.plotting._plotting import *
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.dimred import *
from mito_utils.clustering import *
from mito_utils.classification import *
from mito_utils.plotting_base import *
from mito_utils.heatmaps_plots import *
from matplotlib.gridspec import GridSpec

########################################################################

# Create the parser
my_parser = argparse.ArgumentParser(
    prog='Final models evaluation',
    description=
        """
        Final models evalutaion
        """
)

# Add arguments

# Path_main
my_parser.add_argument(
    '-p', 
    '--path_main', 
    type=str,
    default='..',
    help='Path to samples data. Default: .. .'
)

# Filter
my_parser.add_argument(
    '--sample', 
    type=str,
    default='AML_clones',
    help='Sample to use. Default: AML_clones.'
)

# ntop
my_parser.add_argument(
    '--ntop', 
    type=int,
    default=3,
    help='n top models to evalutate. Default: 3.'
)

# Parse arguments
args = my_parser.parse_args()

##

# Args
# path_main = '/Users/IEO5505/Desktop/example_mito/'
# sample = 'AML_clones'
# n = 3

path_main = args.path_main
sample =  args.sample
n = args.ntop

# Paths
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results/supervised_clones/top_models')
path_clones = os.path.join(path_main, 'results/supervised_clones/reports/report_f1.csv')

##

# Create new folder for sample
make_folder(path_results, sample, overwrite=True)
os.chdir(os.path.join(path_results, sample))  

##  

########################################################################

def main():

    # Read final outs 
    
    # Sample
    L = []
    for x in os.listdir(path_results):
        if bool(re.search('out', x)) and bool(re.search(sample, x)):
            L.append(
                pd.read_csv(os.path.join(path_results, x), index_col=0)
                .assign(top=f'top_{x.split("_")[3]}')
            )
    df_performance = pd.concat(L)

    # Get top_models final results
    d_results = {}
    for x in os.listdir(path_results):
        if bool(re.search('results', x)) and bool(re.search(sample, x)):
            key = f'top_{x.split("_")[3]}'
            with open(os.path.join(path_results, x), 'rb') as f:
                d_results[key] = pickle.load(f)

    ##    
    
    # Iterate over models...
    for top in d_results:
        
        os.chdir(os.path.join(path_results, sample))
        make_folder(os.path.join(path_results, sample), top, overwrite=False)
        os.chdir(os.path.join(path_results, sample, top))
        
        # Get model results
        d_model = d_results[top]
        filtering, dimred, min_cell_number, min_cov_treshold = (
        df_performance
            .query('sample == @sample and top == @top')
            .loc[:, ['filtering', 'dimred', 'min_cell_number', 'min_cov_treshold']]
            .values.tolist()[0]
        )

        # Get AFM
        afm = read_one_sample(path_data, sample=sample)

        # Filter and reduce AFM for viz
        _, a = filter_cells_and_vars(
            afm,
            sample=sample,
            filtering=filtering, 
            min_cell_number=min_cell_number, 
            min_cov_treshold=min_cov_treshold, 
            nproc=8, 
            path_=os.getcwd()
        )
        a = nans_as_zeros(a) # For sklearn APIs compatibility
        X, _ = reduce_dimensions(a, method='UMAP', n_comps=3, sqrt=True)
        embs = pd.DataFrame(X, index=a.obs_names, columns=['UMAP1', 'UMAP2', 'UMAP3'])

        # Sample circle plot
        a.obs['GBC'] = a.obs['GBC'].astype('str')
        df_ = (
            a.obs
            .groupby('GBC').size()
            .to_frame('n_cells')
            .assign(prevalence=lambda x: x['n_cells']/x['n_cells'].sum())
        )
        fig, ax = plt.subplots(figsize=(6,6))
        packed_circle_plot(df_, covariate='prevalence', ax=ax, color='b', annotate=True, fontsize=8)
        fig.savefig('Cicle_packed_plot.png')

        # Sample heatmap
        vois = pd.concat([ 
            rank_clone_variants(a, c, by='perc_ratio', min_clone_perc=0.5, max_perc_rest=0.1) \
            for c in a.obs['GBC'].unique() 
        ]).index

        colors = create_palette(a.obs, 'GBC', 'Set1')
        g = cells_vars_heatmap(
            a[:,vois], cell_anno=[ colors[k] for k in a.obs['GBC'] ],
            anno_colors=colors, heat_label='Heteroplasmy', 
            legend_label='Clones', figsize=(11, 8), title=f'{sample}, {top} model', cbar_position=(0.82, 0.2, 0.02, 0.25),
            title_hjust=0.47, legend_bbox_to_anchor=(0.825, 0.5), legend_loc='lower center', 
            legend_ncol=1, xticks_size=10
        )
        g.fig.savefig(f'{top}_cell_x_vars_heatmap.png')
        

        ##


        # Iterate over clones
        for comparison in d_model.keys():

            # Precision-recall and SHAP 
            d_ = d_model[comparison] # One sample, one model, one clone
            precisions = d_['precisions']
            recalls = d_['recalls']
            tresholds = d_['tresholds']
            alpha = d_['alpha']
            idx_chosen = np.where(tresholds == alpha)[0][0]
            precision, recall, f1, ncells_clone, ncells_sample  = (
                df_performance
                .query('sample == @sample and top == @top and comparison == @comparison')
                .loc[:, ['precision', 'recall', 'f1', 'ncells_clone', 'ncells_sample']]
                .values.tolist()[0]
            )

            ##

            fig, axs = plt.subplots(1,2, figsize=(10.5,5))

            # Pr-recall
            axs[0].plot(recalls, precisions, 'b-', linewidth=2)
            axs[0].plot(recalls[idx_chosen], precisions[idx_chosen], 'ro', markersize=7)
            format_ax(axs[0], title='Precision-recall curve', xlabel='recall', ylabel='precision')
            axs[0].text(0.1, 0.3, f'n cells sample: {int(ncells_sample)}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.25, f'clone prevalence: {ncells_clone / ncells_sample:.2f}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.2, f'precision: {precision:.2f}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.15, f'recall: {recall:.2f}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.1, f'f1: {f1:.2f}', transform=axs[0].transAxes)
            axs[0].spines[['right', 'top']].set_visible(False)
            axs[0].set_xlim((-0.1,1.1))
            axs[0].set_ylim((-0.1,1.1))

            # SHAP
            mean_shap = d_['SHAP'].values.mean(axis=0)
            idx = np.argsort(mean_shap)[::-1]
            df_shap = (
                pd.Series(mean_shap[idx], index=np.array(d_['SHAP'].feature_names)[idx])
                .to_frame('mean_shap')
            )
            stem_plot(pd.concat([df_shap.head(10), df_shap.tail(10)]), 'mean_shap', ax=axs[1])
            format_ax(axs[1], xlabel='mean SHAP', ylabel='variant', title=comparison)
            axs[1].spines[['right', 'top', 'left']].set_visible(False)

            fig.tight_layout()
            fig.savefig(f'{comparison}_feat_importance.png')

            ##

            # Rank variants
            clone = comparison.split('_')[0]
            df_vars = rank_clone_variants(a, clone, min_clone_perc=0.1, max_perc_rest=0.1)
            df_vars['shap_rank'] = np.argsort(df_shap['mean_shap'].sort_values())

            # Visualize top3 SHAP and top3 diff heteroplasmy
            df_ = embs.join([a.obs, pd.DataFrame(a.X, index=a.obs_names, columns=a.var_names)])
            query = f'GBC == "{clone}"'

            # Fig
            fig = plt.figure(figsize=(14,5))
            gs = GridSpec(2, 4, width_ratios=[2, 1, 1, 1], height_ratios=[1, 1])

            # Main
            ax = fig.add_subplot(gs[:, 0])
            draw_embeddings(
                df_, cat='GBC', title=f'Clone {clone}', 
                query=query, 
                ax=ax,
                axes_kwargs={'legend':False}
            )

            ## SHAP
            for i, x in enumerate(df_vars.sort_values('shap_rank').head(3).index):
                ax = fig.add_subplot(gs[0, i+1])
                draw_embeddings(
                    df_, 
                    cont=x,
                    ax=ax,
                    s=7,
                    title=f'% clone: {df_vars.loc[x, "perc_clone"]:.2f}, % rest: {df_vars.loc[x, "perc_rest"]:.2f}',
                    cbar_kwargs={'pos':'outside'}
                )
                ax.axis('off')

            ## Perc_clone / perc_ratio
            for i, x in enumerate(df_vars.sort_values('perc_ratio', ascending=False).head(3).index):
                ax = fig.add_subplot(gs[1, i+1])
                draw_embeddings(
                    df_, 
                    cont=x,
                    s=7,
                    ax=ax,
                    title=f'% clone: {df_vars.loc[x, "perc_clone"]:.2f}, % rest: {df_vars.loc[x, "perc_rest"]:.2f}',
                    cbar_kwargs={'pos':'outside'}
                )
                ax.axis('off')

            fig.tight_layout()
            fig.savefig(f'{comparison}_top_variants.png')

########################################################################

# Run
if __name__ == "__main__":
    main()
    
########################################################################

