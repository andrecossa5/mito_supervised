#!/usr/bin/python

# Top models final assessment

########################################################################

import argparse
import os
import re
import pickle

from sklearn.metrics import auc
from sklearn.metrics import PrecisionRecallDisplay
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.dimred import *
from mito_utils.clustering import *
from mito_utils.classification import *
from mito_utils.plotting_base import *
from mito_utils.embeddings_plots import *
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
# path_main = '/Users/IEO5505/Desktop/mito_bench/'
# sample = 'MDA_PT'
# n = 3

path_main = args.path_main
sample =  args.sample
n = args.ntop

# Paths
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'supervised_clones', 'top_models')
path_clones = os.path.join(path_main, 'results', 'supervised_clones', 'reports', 'report_f1.csv')
path_viz = os.path.join(path_main, 'results', 'supervised_clones', 'viz_top_models')

# Create new folder for sample
make_folder(path_viz, sample, overwrite=True)
os.chdir(os.path.join(path_viz, sample))  

# Colors
samples = samples = [ s for s in os.listdir(path_data) if bool(re.search('AML|MDA', s)) ]
colors_samples = { s:c for s,c in zip(samples, sc.pl.palettes.vega_10[:len(samples)]) }


##


########################################################################

def main():

    # Get top_models final results
    d_results = {}
    L = []
    for x in os.listdir(path_results):
        if bool(re.search('results', x)) and bool(re.search(sample, x)):
            key = f'top_{x.split("_")[3]}'
            with open(os.path.join(path_results, x), 'rb') as f:
                d_results[key] = pickle.load(f)
            L.append(
                pd.concat([ 
                    pd.Series(d_results[key][clone]['performance_dict']) \
                    for clone in d_results[key] 
                ], axis=1).T
                .assign(top=key)
            )
    df_performance = pd.concat(L, axis=0)
    
    ##    
    
    # Iterate over models...
    for top in d_results:
        
        pass
        
        os.chdir(os.path.join(path_viz, sample))
        make_folder(os.path.join(path_viz, sample), top, overwrite=True)
        os.chdir(os.path.join(path_viz, sample, top))
        
        # Get model results
        d_model = d_results[top]
        filtering, dimred, min_cell_number, min_cov_treshold = (
            df_performance
            .query('sample == @sample and top == @top')
            .loc[:, ['filtering', 'dimred', 'min_cell_number', 'min_cov_treshold']]
            .values.tolist()[0]
        )

        # Get AFM
        afm = read_one_sample(path_data, sample=sample, with_GBC=True)

        # Filter and reduce AFM for viz
        _, a = filter_cells_and_vars(
            afm,
            sample=sample,
            filtering=filtering, 
            min_cell_number=min_cell_number, 
            min_cov_treshold=min_cov_treshold, 
            nproc=4, 
            path_=os.path.join(path_viz, sample, top)
        )
        a = nans_as_zeros(a) # For sklearn APIs compatibility
        X, _ = reduce_dimensions(a, method='UMAP', n_comps=2, sqrt=True)
        embs = pd.DataFrame(X, index=a.obs_names, columns=['UMAP1', 'UMAP2'])

        # Sample circle plot
        a.obs['GBC'] = a.obs['GBC'].astype('str')
        df_ = (
            a.obs
            .groupby('GBC').size()
            .to_frame('n_cells')
            .assign(prevalence=lambda x: x['n_cells']/x['n_cells'].sum())
        )
        fig, ax = plt.subplots(figsize=(6,6))
        packed_circle_plot(df_, covariate='prevalence', ax=ax, color=colors_samples[sample], annotate=True, fontsize=8)
        fig.savefig('Cicle_packed_plot.png')

        # Sample heatmap
        vois = pd.concat([ 
            rank_clone_variants(a, c, by='perc_ratio', min_clone_perc=0.5, max_perc_rest=0.1) \
            for c in a.obs['GBC'].unique() 
        ]).index

        # Cell vars hetamap
        colors = create_palette(a.obs, 'GBC', 'Set1')
        g = cells_vars_heatmap(
            a[:,vois], 
            cell_anno='GBC',
            anno_colors=colors, 
            heat_label='Heteroplasmy', 
            legend_label='Clones', 
            figsize=(11, 8), 
            title=f'{sample}, {top} model', 
            cbar_position=(0.82, 0.2, 0.02, 0.25),
            title_hjust=0.47, 
            legend_bbox_to_anchor=(0.825, 0.5), 
            legend_loc='lower center', 
            legend_ncol=1, xticks_size=10,
            order='diagonal_clones'
        )
        g.fig.savefig(f'{top}_cell_x_vars_heatmap.png')
        

        ##


        # Iterate over clones
        for comparison in d_model:

            # Draw and calculate precision-recall curve (and its area) 
            d_ = d_model[comparison] # One sample, one model, one clone
            model = d_['best_estimator']
            X_test = d_['SHAP'].data
            y_test = d_['y_test']
            alpha = d_['alpha']
            
            # Calculate precision recall curve
            pr_curve = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
            precisions = pr_curve.precision
            recalls =  pr_curve.recall

            # Get final precision and recall obtained after picking a decision treshold value 
            # (tuned on the train data) 
            precision, recall, f1, ncells_clone, ncells_sample, model, filtering, dimred  = (
                df_performance
                .query('sample == @sample and top == @top and comparison == @comparison')
                .loc[:, 
                        ['precision', 'recall', 'f1', 'ncells_clone',
                        'ncells_sample', 'model', 'filtering', 'dimred']
                    ]
                .values.tolist()[0]
            )

            ##

            # Viz
            fig, axs = plt.subplots(1,2, figsize=(10.5,5))

            # Pr-recall
            axs[0].plot(recalls, precisions, 'k--', linewidth=1.5)
            axs[0].plot(recall, precision, 'rx', markersize=7)
            format_ax(axs[0], title='Precision-recall curve', xlabel='recall', ylabel='precision')
            axs[0].text(0.1, 0.45, f'Model: {model}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.4, f'Feature_type: {filtering}_{dimred}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.35, f'n cells sample: {int(ncells_sample)}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.3, f'clone prevalence: {ncells_clone / ncells_sample:.2f}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.25, f'precision: {precision:.2f}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.2, f'recall: {recall:.2f}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.15, f'f1: {f1:.2f}', transform=axs[0].transAxes)
            axs[0].text(0.1, 0.1, f'AUC P/Recall: {auc(recalls, precisions):.2f}', transform=axs[0].transAxes)
            axs[0].spines[['right', 'top']].set_visible(False)
            axs[0].set_xlim((-0.1,1.1))
            axs[0].set_ylim((-0.1,1.1))

            # SHAP
            assert (a.var_names == d_['SHAP'].feature_names).all()
            
            df_shap = (
                pd.DataFrame(d_['SHAP'].values, columns=d_['SHAP'].feature_names)
                .mean(axis=0)
                .sort_values(ascending=False)
                .to_frame('mean_shap')
                .assign(shap_rank=lambda x: np.arange(1, x.shape[0]+1))
            )
            stem_plot(pd.concat([df_shap.head(10), df_shap.tail(10)]), 'mean_shap', ax=axs[1])
            format_ax(axs[1], xlabel='mean SHAP', ylabel='variant', title=comparison)
            axs[1].spines[['right', 'top', 'left']].set_visible(False)

            fig.tight_layout()
            fig.savefig(f'{comparison}_feat_importance.png')

            ##

            # Rank variants
            clone = comparison.split('_')[0]
            df_vars = (
                rank_clone_variants(a, clone, min_clone_perc=0, max_perc_rest=1) # No filtering, only ranking
                .join(df_shap)
            )
            
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
                    title=f'Mean SHAP ranking: {i+1}',
                    cbar_kwargs={'pos':'outside'}
                )
                ax.text(.95, .95, f'% +clone: {df_vars.loc[x, "perc_clone"]:.2f}', transform=ax.transAxes, fontsize='xx-small')
                ax.text(.95, .9, f'% +rest: {df_vars.loc[x, "perc_rest"]:.2f}', transform=ax.transAxes, fontsize='xx-small')
                ax.axis('off')

            ## Perc_clone / perc_ratio
            for i, x in enumerate(df_vars.query('perc_clone>=0.5 and perc_rest<=0.25').sort_values('log2FC').head(3).index):
                ax = fig.add_subplot(gs[1, i+1])
                draw_embeddings(
                    df_, 
                    cont=x,
                    s=7,
                    ax=ax,
                    title=f'log2FC AF ranking: {i+1}',  #(% +clone > 0.5, % +rest <0.1)',
                    cbar_kwargs={'pos':'outside'}
                )
                ax.text(.95, .95, f'% + clone: {df_vars.loc[x, "perc_clone"]:.2f}', transform=ax.transAxes, fontsize='xx-small')
                ax.text(.95, .9, f'% + rest: {df_vars.loc[x, "perc_rest"]:.2f}', transform=ax.transAxes, fontsize='xx-small')
                ax.axis('off')

            fig.tight_layout()
            fig.savefig(f'{comparison}_top_variants.png')

########################################################################

# Run
if __name__ == "__main__":
    main()
    
########################################################################

