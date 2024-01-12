"""
Wobble effect and selection in MT-variants.
"""

import os
import sys
import anndata
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.plotting_base import *
from matplotlib.gridspec import GridSpec

##

# Args
# path_main = '/Users/IEO5505/Desktop/mito_bench/'

path_main = sys.argv[1]

##

# Set paths and create folders
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'wobble')

# Read data
MDA_PT = read_one_sample(path_data, 'MDA_PT', with_GBC=True)
MDA_PT = filter_baseline(MDA_PT)
MDA_lung = read_one_sample(path_data, 'MDA_lung', with_GBC=True)
MDA_lung = filter_baseline(MDA_lung)

# Read var_df
var_df = pd.read_csv(
    os.path.join(path_data, 'formatted_table_wobble.csv'),
    index_col=0
)
var_df['Alternative'] = var_df['Variant']
var_df = var_df.drop(columns='Variant')
var_df['Position'] = var_df['Position'].astype('str')
var_df['var'] = var_df['Position'] + '_' + var_df['Reference'] + '>' + var_df['Alternative']
var_df = var_df.set_index('var')

# Merge
anno_vars = list(set(var_df.index) & set(MDA_lung.var_names))
len(anno_vars)

MDA_lung.var = MDA_lung.var.join(var_df.loc[,:])

MDA_lung.var.join(var_df, validate=True)





# Explore!!
df_.groupby('syn_annotation').size()
df_.groupby('Consequence').size()


MDA_lung[:,var_df.index]

MDA_lung.var_names

MDA_lung.var.join(var_df.loc[variants,:], ).reset_index().drop_duplicates()






fig, ax = plt.subplots()
hist(df_.query('type == "to_wobble"'), 'AF', c='b', ax=ax, n=1000)
hist(df_.query('type == "to_WCF"'), 'AF', c='b', ax=ax, n=1000)
ax.set_xlim((0.0, 0.01))

plt.show()


df_.groupby('type').describe()

115/8


