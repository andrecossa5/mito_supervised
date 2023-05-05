#!/usr/bin/python

# Top models retraining, diagnostics and feature importance

########################################################################

# Libraries
import argparse
import sys
import argparse
import pickle
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.dimred import *
from mito_utils.classification import *

# Create the parser
my_parser = argparse.ArgumentParser(
    prog='Top models evaluation',
    description=
        """
        Systematically testing the ability of (filtered) MT-SNVs to distinguish
        ground truth clonal labels from lentiviral barcoding.
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
    default='MDA_clones',
    help='Sample to use. Default: MDA_clones.'
)

# ncombos
my_parser.add_argument(
    '--ncombos', 
    type=int,
    default=100,
    help='n combinations of hyperparameters to test. Default: 100.'
)

# ntop
my_parser.add_argument(
    '--ntop', 
    type=int,
    default=3,
    help='n top models to evalutate. Default: 3.'
)

# ncores
my_parser.add_argument(
    '--ncores', 
    type=int,
    default=8,
    help='ncores to use for model training. Default: 8.'
)

# Score
my_parser.add_argument(
    '--score', 
    type=str,
    default='f1',
    help='Classification performance scoring type. Default: f1-score.'
)

# Parse arguments
args = my_parser.parse_args()

##

# Args
# path_main = '/Users/IEO5505/Desktop/example_mito/'
# sample = 'AML_clones'
# ncores = 8
# score = 'f1'
# n_combos = 5
# n = 3

path_main = args.path_main
sample =  args.sample
ncores = args.ncores
score = args.score
n_combos = args.ncombos
n = args.ntop

# Paths
path_data = os.path.join(path_main, '/data/')
path_results = os.path.join(path_main, 'results/supervised_clones/top_models/')
path_clones = os.path.join(path_main, 'results/supervised_clones/reports/report_f1.csv')

##

########################################################################

def main():
    
    # Logger
    logger = set_logger(path_results, f'log_top_{sample}.txt')
    logger.info(
        f""" 
        Execute top models final evaluation: \n
        --sample {sample} 
        --ncombos {n_combos} 
        --score {score}
        --ntop {n}
        """
    )
    
    T = Timer()
    T.start()

    # Data    
    afm = read_one_sample(path_data, sample=sample)
    
    # Get models 
    clones = pd.read_csv(path_clones, index_col=0)
    top_models = (
        clones
        .query('sample == @sample')
        .assign(job=lambda x: x['filtering'] + '|' + x['dimred'] + '|' + x['model'] + '|' + x['tuning'])
        .groupby('job')
        .agg('mean', numeric_only=True)
        .sort_values('f1', ascending=False)
        .head(n)
    )
    
    clones.shape
    top_models = top_models.iloc[:, 5:]
    
    top_dict = { }
    for i in range(n):
        s = top_models.iloc[i,:]
        filtering, dimred, model, tuning = s.name.split('|')
        d_model = {
            'filtering' : filtering,
            'dimred' : dimred,
            'model' : model,
            'tuning' : tuning,
            'min_cell_number' : s['min_cell_number'],
            'min_cov_treshold' : s['min_cov_treshold']
        }
        top_dict[f'top_{i+1}_{sample}'] = d_model
        
        
    ##
    
    
    # Here we go
    for job in top_dict:
        
        # Get options
        model_options = top_dict[job]
        filtering = model_options['filtering']
        dimred = model_options['dimred']
        min_cell_number = model_options['min_cell_number']
        min_cov_treshold = model_options['min_cov_treshold']
        model = model_options['model']
        GS_mode = model_options['tuning']
        
        # Filters
        if dimred == 'no_dimred':

            _, a = filter_cells_and_vars(
                afm,
                # blacklist=blacklist,
                sample=sample,
                filtering=filtering, 
                min_cell_number=min_cell_number, 
                min_cov_treshold=min_cov_treshold, 
                nproc=ncores, 
                path_=path
            )

            # Extract X, y
            a = nans_as_zeros(a) # For sklearn APIs compatibility
            ncells = a.shape[0]
            n_clones_analyzed = len(a.obs['GBC'].unique())
            X = a.X
            y = pd.Categorical(a.obs['GBC'])
            Y = one_hot_from_labels(y)

        else:

            _, a = filter_cells_and_vars(
                afm,
                sample=sample,
                filtering=filtering, 
                min_cell_number=min_cell_number, 
                min_cov_treshold=min_cov_treshold, 
                nproc=ncores
            )

            # Extract X, y
            a = nans_as_zeros(a) # For sklearn APIs compatibility
            ncells = a.shape[0]
            n_clones_analyzed = len(a.obs['GBC'].unique())
            X, _ = reduce_dimensions(a, method=dimred, n_comps=30, sqrt=False)
            y = pd.Categorical(a.obs['GBC'])
            Y = one_hot_from_labels(y)

        # One per clone
        L_performance = []
        d_results = {}
        
        # Here we go
        for i in range(Y.shape[1]):  
            
            t = Timer()
            t.start()
            
            comparison = f'{y.categories[i]}_vs_rest'
            y_ = Y[:,i]
            
            logger.info(f'Starting comparison {comparison} ({i+1}/{Y.shape[1]})...')

            # Classification
            results = classification(
                X, y_, key=model, GS=True, GS_mode=GS_mode,
                score=score, n_combos=n_combos, cores_model=ncores, cores_GS=1,
                full_output=True, feature_names=a.var_names
            )
            
            # Performance dict
            results['performance_dict'] |= {
                'sample' : sample,
                'filtering' : filtering, 
                'dimred' : dimred,
                'min_cell_number' : min_cell_number,
                'min_cov_treshold' : min_cov_treshold,
                'ncells_clone' : y_.sum(),
                'ncells_sample' : ncells,
                'n_clones_analyzed' : n_clones_analyzed,
                'n_features' : X.shape[1],
                'model' : model,
                'score_for_tuning' : score,
                'comparison' : comparison
            }         
            L_performance.append(results['performance_dict'])
            
            # Results
            d_results[comparison] = results
            logger.info(f'Finished {comparison} ({i+1}/{Y.shape[1]}): {t.stop()}')
            
        ##
            
        # Save all results
        df = pd.DataFrame(L_performance)
        print(df['f1'].describe())
            
        # Save df_performance
        df.to_csv(
            os.path.join(
                path_results, 
                f'out_top_{job}_{sample}.csv'
            )
        )
        
        # Save pickled
        with open(os.path.join(path_results, f'results_top_{job}_{sample}.pkl'), 'wb') as f:
            pickle.dump(d_results, f)
            
#######################################################################

# Run program
if __name__ == "__main__":
    main()

#######################################################################