"""
Visualization of clones and samples classification performances.
"""

# Code
import os
import sys
from mito_utils.preprocessing import *
from mito_utils.diagnostic_plots import *
from mito_utils.heatmaps_plots import *
from mito_utils.utils import *
from mito_utils.plotting_base import *
from matplotlib.gridspec import GridSpec


##


# Args
# sample = 'MDA_lung'
# path_main = '/Users/IEO5505/Desktop/mito_bench/'

path_main = sys.argv[1]
sample = sys.argv[2]


# Set paths and create folders
path_data = os.path.join(path_main, 'data')
path_output = os.path.join(path_main, 'results', 'supervised_clones', 'output')
path_report = os.path.join(path_main, 'results', 'supervised_clones', 'reports')

# Create sample folder for viz
make_folder(os.path.join(path_main, 'results', 'supervised_clones', 'variants_characterization'), sample)
path_viz = os.path.join(path_main, 'results', 'supervised_clones', 'variants_characterization', sample)

# Read data
afm = read_one_sample(path_data, sample)
clones = pd.read_csv(os.path.join(path_report, 'report_f1.csv'), index_col=0)


##


##############

# Variants biological anno: EDepaquale lab
fun_df = pd.read_csv(os.path.join(path_data, 'rev_table.txt'), sep='\t')
variants = pd.Series(filter_baseline(afm).var_names).to_frame('var')
fun_df['var'] = fun_df['Position'].astype('str') + '_' + fun_df['Reference'] + '>' + fun_df['Variant'] 
                
# Format
df_annot = fun_df[fun_df['var'].isin(variants['var'])]
df_annot = (
    df_annot.loc[:, [
        'Position', 'Reference', 'Variant', 'Consequence', 'Symbol', \
        'Biotype', 'SIFT', 'PolyPhen', 'Disease', 'Status'
    ]]
    .assign(var=lambda x: 
       x['Position'].astype('str') + '_' + x['Reference'] + '>' + x['Variant']
    )
    .drop(columns=['Position', 'Reference', 'Variant'])
    .set_index('var')
)

# Take out positive cells per var info
df_positive = (
    pd.Series(np.sum(afm.X > 0, axis=0), index=afm.var_names)
    .to_frame('n_positive')
    .assign(variant_type=lambda x: x.index.map(lambda x: x.split('_')[1]))
    .loc[lambda x: ~x['variant_type'].str.contains('N>')]
)
order = (
    df_positive.groupby('variant_type')
    .agg('mean')
    .sort_values('n_positive', ascending=False)
    .index
)

# Fig
fig = plt.figure(figsize=(13, 4))
gs = GridSpec(1, 3, figure=fig, width_ratios=[5, 8, 3])

# Subs type
ax = fig.add_subplot(gs[0,0])
strip(df_positive, 'variant_type', 'n_positive', c='#2e2d2d', s=3, order=order, ax=ax)
ax.spines[['top', 'right', 'left']].set_visible(False)
medians = (
    df_positive.groupby('variant_type')
    .agg('mean')
    .sort_values('n_positive', ascending=False)
    .values
)
for i, y in enumerate(medians):
    ax.hlines(y, i-.25, i+.25, 'r', zorder=4)
format_ax(ax, title=f'n of +events, by substitution type', ylabel='n cells', rotx=90)

##

# Gene 
x = 'Symbol'
ax = fig.add_subplot(gs[0,1])
_ = df_annot[x].value_counts().to_frame('count')
bar(_, 'count', c='lightgrey', edgecolor='k', ax=ax)
format_ax(ax, title='MT-SNVs by genomic feature', ylabel='n variants', xticks=_.index.values, 
        rotx=90 if _.index.size>5 else 0)
ax.spines[['top', 'right', 'left']].set_visible(False)

##

# Biotype
x = 'Biotype'
ax = fig.add_subplot(gs[0,2])
_ = df_annot[x].value_counts().to_frame('count')
bar(_, 'count', c='lightgrey', edgecolor='k', ax=ax)
format_ax(ax, title='MT-SNVs by biotype', ylabel='n variants', xticks=_.index.values, rotx=90)
ax.spines[['top', 'right', 'left']].set_visible(False)

# Plot
fig.tight_layout()
fig.savefig(os.path.join(path_viz, 'variants_characterization_I.png'))


##


# Functional anno

# Fig
fig = plt.figure(figsize=(13, 4))
gs = GridSpec(1, 3, figure=fig, width_ratios=[7, 5, 5])

# Codon
_ = df_annot['Consequence'].value_counts().to_frame('count')

d_conversion = {
    
    'missense_variant' : 'missense',
    'synonymous_variant' : 'synonimous',
    'non_coding_transcript_exon_variant' : 'non_coding',
    'stop_gained' : 'stop_gain',
    'incomplete_terminal_codon_variant,coding_sequence_variant' : 'incomplete_terminal',
    'intergenic_variant' : 'intergenic',
    'stop_retained_variant' : 'stop_retained'

}

# Convert names for better plotting
_.index = [ d_conversion[x] if x in d_conversion else x for x in _.index ]

# Axes
ax = fig.add_subplot(gs[0,0])
bar(_, 'count', c='lightgrey', edgecolor='k', ax=ax)
format_ax(ax, title='MT-SNVs codon effect', ylabel='n variants', xticks=_.index.values, rotx=90)
ax.spines[['top', 'right', 'left']].set_visible(False)

# SIFT
_ = df_annot['SIFT'].map(lambda x: x.split('(')[0]).value_counts().to_frame('count')
ax = fig.add_subplot(gs[0,1])
bar(_, 'count', c='lightgrey', edgecolor='k', ax=ax)
format_ax(ax, title='Protein effect (SIFT)', ylabel='n variants', xticks=_.index.values, rotx=90)
ax.spines[['top', 'right', 'left']].set_visible(False)

# Polyphen
_ = df_annot['PolyPhen'].map(lambda x: x.split('(')[0]).value_counts().to_frame('count')
ax = fig.add_subplot(gs[0,2])
bar(_, 'count', c='lightgrey', edgecolor='k', ax=ax)
format_ax(ax, title='Protein effect (PolyPhen)', ylabel='n variants', xticks=_.index.values, rotx=90)
ax.spines[['top', 'right', 'left']].set_visible(False)

# Plot
fig.tight_layout()
fig.savefig(os.path.join(path_viz, 'variants_characterization_II.png'))

##############


##


############## 

# Are homoplasmic, high-prevalence variants under active selection??
var_prevalence = np.sum(afm.X>0, axis=0) / afm.shape[0]
var_mean_AF = np.nanmean(afm.X, axis=0)
df_ = (
    pd.DataFrame(
        {'prevalence' : var_prevalence, 'mean_AF': var_mean_AF},
        index=afm.var_names
    )
    .assign(type=lambda x: np.where((x['prevalence']>0.5) & (x['mean_AF']>0.5), 'high', 'low'))
    .join(df_annot.loc[:, ['Consequence', 'SIFT', 'PolyPhen']], how='inner') 
    # NB. Make sure df_annot is filtered!
)
n_high = df_.query('type == "high"').shape[0]

##

# Plot
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, figure=fig, width_ratios=[2,2])

# Biplot mean_AF/prevalence
ax = fig.add_subplot(gs[0,0])
ax.plot(df_.query('type == "low"')['mean_AF'], df_.query('type == "low"')['prevalence'], 'k.', markersize=3)
ax.plot(df_.query('type == "high"')['mean_AF'], df_.query('type == "high"')['prevalence'], 'r+', markersize=6)
format_ax(ax, xlabel='Mean AF', ylabel='Prevalence')

ax.text(.4, .2, f'MT-SNVs:', transform=ax.transAxes)
ax.text(.4, .15, f'-{df_.shape[0]} passing baseline filters', transform=ax.transAxes)
ax.text(.4, .1, f'-{n_high} >0.5 mean_AF and prevalence',  transform=ax.transAxes)
ax.spines[['right', 'top']].set_visible(False)

# Lollipop consequence
ax = fig.add_subplot(gs[0,1])
stem_plot(df_.query('type == "high"').sort_values('mean_AF', ascending=False).head(10), 'mean_AF', ax=ax)
format_ax(ax, xlabel='Mean AF')
ax.spines[['right', 'top', 'left']].set_visible(False)

tests = [
    df_['Consequence'] == 'non_coding_transcript_exon_variant',
    df_['Consequence'] == 'synonymous_variant',
    df_['Consequence'] == 'missense_variant'
]
df_['codon'] = np.select(tests, ['non coding', 'synonimous', 'missense'])
annot_list = (
    df_.query('type == "high"')
    .sort_values('mean_AF', ascending=False)
    .loc[:, ['codon', 'PolyPhen']].values.tolist()[:10]
)

_ = .97
i = 0
vep, poly = annot_list[i]
#print(i, vep, poly)
ax.text(.1, _, f'VEP: {vep}; PolyPhen: {poly}', transform=ax.transAxes)
for x in annot_list[1:]:
    i += 1
    vep, poly = x
    ax.text(.1, _-(0.1*i), f'VEP: {vep}; PolyPhen: {poly}', transform=ax.transAxes)

# Save
fig.tight_layout()
fig.savefig(os.path.join(path_viz, 'variants_selection_evidence.png'))

##############


##


############## 
