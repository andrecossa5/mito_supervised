"""
Script to investigate the dynamics of PT-lungs MT-SNVs
"""

# Code
import matplotlib
from mito_utils.preprocessing import *
from mito_utils.phylo import *
from mito_utils.plotting_base import *
from mito_utils.diagnostic_plots import *
from matplotlib.gridspec import GridSpec
matplotlib.use('macOSX')


##


# Read AFM and filter vars
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'supervised_clones', 'visualization')

# Filter AFM
afm_PT = read_one_sample(path_data, 'MDA_PT', with_GBC=True)
afm_lung = read_one_sample(path_data, 'MDA_lung', with_GBC=True)

PT_clones = afm_PT.obs['GBC'].value_counts().loc[lambda x: x>10].index
lung_clones = afm_lung.obs['GBC'].value_counts().loc[lambda x: x>10].index

# Old top 
# old_top = ['CGGGAGCAGGACAGCGAC', 'GGACAGTGGAAGCAAGGG', 'GATGTAATTTGTTATAGC']

# New top
top_met = list(set(PT_clones) & set(lung_clones))
# d_rev = {'A':'T', 'G':'C', 'T':'A', 'C':'G', 'N':'N'}
# top_met_rev = [ ''.join([ d_rev[x] for x in reversed(x) ]) for x in top_met ]
# [ x in old_top for x in top_met_rev ]

# Frequencies
afm_PT.obs['GBC'].value_counts().loc[top_met]
afm_lung.obs['GBC'].value_counts().loc[top_met]

# Take out only top_met clones, and GT filter
_, PT = filter_cells_and_vars(afm_PT, cells=afm_PT.obs.query('GBC in @top_met').index)
_, PT = filter_afm_with_gt(PT, min_cells_clone=5)
_, lung = filter_cells_and_vars(afm_lung, cells=afm_lung.obs.query('GBC in @top_met').index)
_, lung = filter_afm_with_gt(lung, min_cells_clone=5)



PT



##


pd.Series(AFM_to_seqs(PT)).value_counts()
pd.Series(AFM_to_seqs(lung)).value_counts()


new_vars = summary_stats_vars(
    lung, lung.var_names[~lung.var_names.isin(PT.var_names)]
).sort_values('median_AF', ascending=False).head(5).index


agg_new = lung.obs.loc[:, ['GBC']].join(
    pd.DataFrame(
        nans_as_zeros(lung[:,new_vars]).X, 
        index=lung.obs_names, 
        columns=new_vars
)).groupby('GBC').mean().T

np.sum(agg_new>0, axis=1)


##


clone = top_met[2]
rank_clone_variants(
    lung, group=clone, rank_by='custom_perc_tresholds',
    min_clone_perc=.75, max_perc_rest=.25
)




