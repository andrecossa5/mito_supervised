"""
QC MT-library on a single sample.
"""

import os
from mito_utils.preprocessing import *
from mito_utils.diagnostic_plots import *
matplotlib.use('macOSX')


##

# Paths
path_main = '/Users/IEO5505/Desktop/mito_bench/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'sample_diagnostics')

# Sample
sample = 'AML_clones'

# Read AFM, cells_meta and MT_gene expression
# AFM = sc.read(os.path.join(path_data, sample, 'AFM.h5ad'))
afm = read_one_sample(path_data, sample=sample, with_GBC=True)
meta = pd.read_csv(os.path.join(path_data, 'cells_meta.csv'), index_col=0)
afm.obs = afm.obs.join(meta.query('sample==@sample')[['nUMIs', 'mito_perc']])
mt_expr = (
    pd.read_csv(
        os.path.join(path_data, 'miscellanea', 'mt_genes_expr.csv'), index_col=0
    ).loc[afm.obs_names]
)


##


# 1. n consensus UMIs (Q30) per median UMI group size (across cells)

# ...

# 2. AF distribution all variants (unfiltered)
fig, ax = plt.subplots(figsize=(4.5,4.5))
vars_AF_dist(afm, ax=ax, color='k', title='')
fig.savefig(os.path.join(path_results, f'{sample}_AF_spectrum.png'), dpi=500)

# 3. MT-coverage by gene plot
fig, ax = plt.subplots(figsize=(4.5,4.5), subplot_kw={'projection': 'polar'})
MT_coverage_by_gene_polar(afm, ax=ax, title='')
fig.tight_layout()
fig.savefig(os.path.join(path_results, 'MT_coverage.png'), dpi=300)

# 4. MT-gene mean expression vs MAESTER mean base coverage
mean_expr = mt_expr.mean(axis=0)

# Annotate MT-genome sites
mt_genes_positions = [ x for x in all_mt_genes_positions if x[0] in mt_expr.columns ]
sites = afm.uns['per_position_coverage'].columns
annot = {}
for x in sites:
    x = int(x)
    mapped = False
    for mt_gene, start, end in mt_genes_positions:
        if x>=start and x<=end:
            annot[str(x)] = mt_gene
            mapped = True
    if not mapped:
        annot[str(x)] = 'other'
mean_site_cov = afm.uns['per_position_coverage'].T.mean(axis=1).to_frame('cov')
mean_site_cov['gene'] = pd.Series(annot)
mean_site_cov = mean_site_cov.query('gene!="other"').groupby('gene')['cov'].mean()
mean_site_cov = mean_site_cov[mean_expr.index]

##

# Fig
fig, ax = plt.subplots(figsize=(4.5,4.5))
ax.plot(mean_expr.values, mean_site_cov.values, 'ko')
sns.regplot(data=pd.DataFrame({'expr':mean_expr, 'cov':mean_site_cov}), 
            x='expr', y='cov', ax=ax, scatter=False)
format_ax(ax, title='MT-transcripts counts vs site coverage',
          xlabel='Mean expression (gene nUMI, 10x)', 
          ylabel='Mean site coverage (per-site nUMI, MAESTER)')
corr = np.corrcoef(mean_expr.values, mean_site_cov.values)[0,1]
ax.text(.05, .9, f'Pearson\'s r: {corr:.2f}', transform=ax.transAxes)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'{sample}_site_coverage_by_gene_expression.png'), dpi=500)


##

