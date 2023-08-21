"""
Script to find the ground truth variants of a population with gt clones, at different 
tresholds of "exclusivity".
"""

import os
import sys
import warnings
import matplotlib
from mito_utils.preprocessing import *
from mito_utils.clustering import *
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.plotting_base import *
warnings.filterwarnings('ignore')


##


# Args
# sample = 'AML_clones'
# path_main = '/Users/IEO5505/Desktop/mito_bench'
path_main = sys.argv[1]
sample = sys.argv[2]


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


def main():


    def process_one_sample(sample, t_min=.75, t_max=.25):

        d = {}

        # Load data
        afm = read_one_sample(path_data, sample, with_GBC=True)

        # Here we go
        a_cells = filter_cells_coverage(afm)
        a_cells = filter_baseline(a_cells)
        df_gt = summary_stats_vars(a_cells)

        # Supervised search
        gt_l = [
            rank_clone_variants(
                a_cells, var='GBC', group=g, rank_by='custom_perc_tresholds',
                min_clone_perc=t_min, max_perc_rest=t_max
            ).assign(clone=g)
            for g in a_cells.obs['GBC'].unique()
        ]
        vois_df = (
            pd.concat(gt_l)
            .join(df_gt)
            .query('n_cells_clone>10')
            .sort_values('log2_perc_ratio', ascending=False)
        )
        vois = vois_df.index.unique()

        # How many clones and cells do we loose, if we would like to retain only cells 
        # from distinguishable clones?
        d['all_cells'] = a_cells.shape[0]
        d['all_clones'] = a_cells.obs['GBC'].unique().size
        clones_10_cells = a_cells.obs.groupby('GBC').size().loc[lambda x:x>10].index
        d['clones_10_cells'] = clones_10_cells.size
        clones_recoverable = vois_df.query('n_cells_clone>10')['clone'].unique()
        d['clones_recoverable'] = clones_recoverable.size
        d['cells_clones_10_cells'] = a_cells.obs.loc[
            a_cells.obs['GBC'].isin(clones_10_cells)].shape[0]
        d['cells_clones_recoverable'] = a_cells.obs.loc[
            a_cells.obs['GBC'].isin(clones_recoverable)].shape[0]
        
        # How many variants do we have for each recoverable GT clone (10> cells)??
        d['n_vars_baseline'] = a_cells.shape[1]
        d['n_exclusive_vars'] = vois.size
        d['median_n_vars'] = np.median(
            vois_df
            .groupby('clone')
            .size().sort_values(ascending=False)
        )
        d['t'] = (t_min, t_max)

        return d, vois, clones_recoverable
    

    ##


    # Process
    t_list = [(.95,.05), (.85,.15), (.75,.25)]
    L = []
    vois_d = {}

    for x, y in t_list:
        d, vois, good_clones = process_one_sample(sample, t_min=x, t_max=y)
        if x == .75:
            (
                pd.DataFrame(good_clones, columns=['clone'])
                .to_csv(os.path.join(path_data, sample, 'good_clones.csv'))
            )
        L.append(d)
        vois_d[f'({x},{y})'] = vois

    # Save GT vois
    with open(os.path.join(path_tmp, f'{sample}_GT_variants.pickle'), 'wb') as f:
        pickle.dump(vois_d, f)

    # Viz
    df_ = pd.DataFrame(L)

    fig, axs =  plt.subplots(1,3,figsize=(12,4))
    bar(df_, x='t', y='n_exclusive_vars', ax=axs[0], c='lightgrey', edgecolor='k', s=.75)
    format_ax(
        axs[0],
        title=f'GT variants ({df_["n_vars_baseline"].unique()[0]} at baseline)',
        xlabel='tresholds', ylabel='n MT-SNVs', xticks=df_['t'].values,
        reduced_spines=True
    )
    bar(df_, x='t', y='clones_recoverable', ax=axs[1], c='lightgrey', edgecolor='k', s=.75)
    format_ax(
        axs[1],
        title=f'"Recoverable" clones ({df_["clones_10_cells"].unique()[0]} major clones)',
        xlabel='tresholds', ylabel='n clones', xticks=df_['t'].values,
        reduced_spines=True
    )
    bar(df_, x='t', y='cells_clones_recoverable', ax=axs[2], c='lightgrey', edgecolor='k', s=.75)
    format_ax(
        axs[2],
        title=f'"Recoverable" cells ({df_["cells_clones_10_cells"].unique()[0]} major clones)',
        xlabel='tresholds', ylabel='n cells', xticks=df_['t'].values,
        reduced_spines=True
    )

    # Save fig
    fig.tight_layout()
    fig.savefig(
        os.path.join(path_viz, f'{sample}_GT_variants_and_recoverable_clones.png'),
        dpi=300
    )

###############################################

# Run
if __name__ == '__main__':
    main()


