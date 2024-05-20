"""
Exploratory analysis and checks on the three original and formatted AFMs.
"""

import os
import re
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mito_utils.preprocessing import *
from mito_utils.diagnostic_plots import *
matplotlib.use('macOSX')


##

# Paths
path_main = '/Users/IEO5505/Desktop/mito_bench/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'sample_diagnostics')

# Read data in a dictionary
samples = [ s for s in os.listdir(path_data) if bool(re.search('AML|MDA', s)) ]
AFMs = { s : read_one_sample(path_data, sample=s) for s in samples }
n = len(AFMs)

#

################################################################

# Cell level diagnostics

############## n covered positions/total position. Cell distribution.

fig, axs = plt.subplots(1,n,figsize=(4*n,4.2),constrained_layout=True)
for i, x in enumerate(samples):
    cell_n_sites_covered_dist(AFMs[x], ax=axs[i], title=x)
fig.suptitle('n sites covered (total=16569) across cells')
fig.savefig(os.path.join(path_results, 'covered_sites_per_cell.png'), dpi=300)

############## n covered variants/total variants. Cell distribution.

fig, axs = plt.subplots(1,n,figsize=(4*n,4.2),constrained_layout=True)
for i, x in enumerate(samples):
    cell_n_vars_detected_dist(AFMs[x], ax=axs[i], title=x)
fig.suptitle('n total raw MT-SNVs calls (total=16569*3), across cells')
fig.savefig(os.path.join(path_results, 'variants_per_cell.png'), dpi=300)

############## Median site quality per cell distribution

fig, axs = plt.subplots(1,n,figsize=(4*n,4.2),constrained_layout=True)
for i, x in enumerate(samples):
    mean_site_quality_cell_dist(AFMs[x], ax=axs[i], title=x)
fig.suptitle('Mean site quality, across cells')
fig.savefig(os.path.join(path_results, 'mean_site_quality_per_cell.png'), dpi=300)

################################################################

# Position level diagnostics

############## Per position mean (over cells) coverage

fig, axs = plt.subplots(1,n,figsize=(4*n,4.2),constrained_layout=True)
for i, x in enumerate(samples):
    mean_position_coverage_dist(AFMs[x], ax=axs[i], title=x, xlim=(-200,1800))
fig.suptitle('Mean position coverage, across cells')
fig.savefig(os.path.join(path_results, 'mean_coverage_per_position.png'), dpi=300)

############## Mean base quality per position 

fig, axs = plt.subplots(1,n,figsize=(4*n,4.2),constrained_layout=True)
for i, x in enumerate(samples):
    mean_position_quality_dist(AFMs[x], ax=axs[i], title=x)
fig.suptitle('Mean position quality, across cells')
fig.savefig(os.path.join(path_results, 'mean_quality_per_position.png'), dpi=300)

################################################################

# Variant level diagnostics

############## % of cells in which a variant is detected: distribution

fig, axs = plt.subplots(1,n,figsize=(4*n,4.2),constrained_layout=True)
for i, x in enumerate(samples):
    vars_n_positive_dist(AFMs[x], ax=axs[i], title=x, xlim=(-80,200))
fig.suptitle('n of positive cells, per variant')
fig.savefig(os.path.join(path_results, 'cells_per_variant.png'), dpi=300)

############## Ranked AF distributions (VG-like)

fig, axs = plt.subplots(1,n,figsize=(4*n,4.2),constrained_layout=True)
for i, x in enumerate(samples):
    vars_AF_dist(AFMs[x], ax=axs[i], color='k', title=x)
fig.suptitle('Ranked variant AFs')
fig.savefig(os.path.join(path_results, 'VG_like_AFs.png'), dpi=300)

# Fig paper
fig, ax = plt.subplots(figsize=(4.5,4.5))
vars_AF_dist(AFMs['MDA_clones'], ax=ax, color='k')
ax.set(title='')
fig.savefig(os.path.join(path_results, 'VG_like_AFs.pdf'), dpi=500)

##############

# Fancy coverage plot
fig, axs = plt.subplots(1,n,figsize=(4*n,4.2), subplot_kw={'projection': 'polar'})
for i, x in enumerate(samples):
    MT_coverage_polar(AFMs[x], ax=axs[i], title=x)
fig.suptitle('MT-genome coverage')
fig.tight_layout()
fig.savefig(os.path.join(path_results, 'MT_coverage.png'), dpi=300)

# Fig paper
# fig, ax = plt.subplots(figsize=(4.5,4.5), subplot_kw={'projection': 'polar'})
# MT_coverage_polar(AFMs['MDA_clones'], ax=ax)
# fig.tight_layout()
# fig.savefig(os.path.join(path_results, 'MT_coverage.pdf'), dpi=500)

##############

# Circle packed plot, one fig per sample (in vitro)
fig, axs = plt.subplots(2, 1, figsize=(5,9))

s = 'MDA_clones'
df_ = (
    AFMs[s].obs
    .groupby('GBC')
    .size()
    .to_frame('n_cells')
    .assign(prevalence=lambda x: x['n_cells'] / x['n_cells'].sum())
)
packed_circle_plot(df_, covariate='prevalence', ax=axs[0], color=colors[s], 
                annotate=True, fontsize=9, alpha=0.35, linewidth=1.5, annot_treshold=0.01)
format_ax(axs[0], title=s, title_size=10)

s = 'AML_clones'
df_ = (
    AFMs[s].obs
    .groupby('GBC')
    .size()
    .to_frame('n_cells')
    .assign(prevalence=lambda x: x['n_cells'] / x['n_cells'].sum())
)
packed_circle_plot(df_, covariate='prevalence', ax=axs[1], color=colors[s], 
                annotate=True, fontsize=9, alpha=0.35, linewidth=1.7, annot_treshold=0.01)
format_ax(axs[1], title=s, title_size=10)
        
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.85, hspace=0.3)

plt.show()
fig.savefig(os.path.join(path_results, f'in_vitro_circle_packed.png'))

##############


##