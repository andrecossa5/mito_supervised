#!/usr/bin/python

# Clones classification script

########################################################################

# Libraries
import sys 
import argparse

# Create the parser
my_parser = argparse.ArgumentParser(
    prog='clones_classification',
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
    '--path_data', 
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

# Filter
my_parser.add_argument(
    '--filtering_key', 
    type=str,
    default='MI_TO',
    help='Method to filter MT-SNVs. Default: ludwig2019.'
)

# Filter
my_parser.add_argument(
    '--path_filtering', 
    type=str,
    default='MI_TO',
    help='Path to filtering options json. Default: None.'
)

# Priors
my_parser.add_argument(
    '--path_priors', 
    type=str,
    default='None',
    help='Path to priors.csv. Default: None.'
)

# Dimred
my_parser.add_argument(
    '--dimred', 
    type=str,
    default='no_dimred',
    help='Method to reduce the dimension of the SNVs space (top 1000 SNVs selected) by pegasus. Default: no_dimred.'
)

# Dimred
my_parser.add_argument(
    '--n_comps', 
    type=int,
    default=30,
    help='n of components in the dimensionality reduction step. Default: 30.'
)

# Model
my_parser.add_argument(
    '--model', 
    type=str,
    default='xgboost',
    help='Classifier chosen. Default: xgboost.'
)

# ncombos
my_parser.add_argument(
    '--ncombos', 
    type=int,
    default=50,
    help='n combinations of hyperparameters to test. Default: 50.'
)

# ncores
my_parser.add_argument(
    '--ncores', 
    type=int,
    default=8,
    help='ncores to use for model training. Default: 8.'
)

# GS_mode
my_parser.add_argument(
    '--GS_mode', 
    type=str,
    default='random',
    help='Type of hyperparameters tuning. Default: random. Alternative: bayes.'
)

# min_cell_number
my_parser.add_argument(
    '--min_cell_number', 
    type=int,
    default=10,
    help='Include in the analysis only cells with membership in clones with >= min_cell_number. Default: 10.'
)

# min_cov_treshold
my_parser.add_argument(
    '--min_cov_treshold', 
    type=int,
    default=50,
    help='Include in the analysis only cells MAESTER sites mean coverage > min_cov_treshold. Default: 30.'
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

path_data = args.path_data
path_priors = args.path_priors
sample = args.sample
dimred = args.dimred
filtering_key = args.filtering_key
path_filtering = args.path_filtering
model = args.model
n_combos = args.ncombos
ncores = args.ncores 
score = args.score
min_cell_number = args.min_cell_number
n_comps = args.n_comps
GS_mode = args.GS_mode


##


# Code
import json
import pickle
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.dimred import *
from mito_utils.classification import *

# Set logger
path = os.getcwd() 
logger = set_logger(path, f'log_{sample}_{filtering_key}_{dimred}_{model}_{GS_mode}_{min_cell_number}.txt')


##


########################################################################

# Main
def main():

    T = Timer()
    T.start()

    # Load data
    t = Timer()
    t.start()

    logger.info(
        f""" 
        Execute clones classification: \n
        --sample {sample} 
        --filtering_key {filtering_key} 
        --path_filtering {path_filtering} 
        --dimred {dimred} 
        --model {model}
        --ncombos {n_combos} 
        --score {score} 
        --min_cell_number {min_cell_number}
        """
    )
    
    # Read AFM
    afm = read_one_sample(path_data, sample=sample, with_GBC=True)
    ncells0 = afm.shape[0]
    n_all_clones = len(afm.obs['GBC'].unique())

    # Prep filtering kwargs
    with open(path_filtering, 'r') as file:
        FILTERING_OPTIONS = json.load(file)

    if filtering_key in FILTERING_OPTIONS:
        d = FILTERING_OPTIONS[filtering_key]
        filtering = d['filtering']
        filtering_kwargs = d['filtering_kwargs'] if 'filtering_kwargs' in d else {}
    else:
        raise KeyError(f'{filtering_key} not in {path_filtering}!')

    # Filter variants
    _, a = filter_cells_and_vars(
        afm, 
        sample_name=sample,
        filtering=filtering, 
        min_cell_number=min_cell_number,
        nproc=ncores,
        filtering_kwargs=filtering_kwargs,
        lineage_column='GBC',
        path_priors=path_priors if path_priors != 'NULL' else None
    )


    ##


    # Re-format
    a = nans_as_zeros(a) # For sklearn APIs compatibility
    cells = a.obs_names
    variants = a.var_names
    ncells = cells.size
    n_clones_analyzed = len(a.obs['GBC'].unique())
    
    # Filter 'good quality' cells and variants, reduce dimension if necessary
    if dimred == 'no_dimred':
        X = a.X
        y = pd.Categorical(a.obs['GBC'])
        Y = one_hot_from_labels(y)
    else:
        X, _ = reduce_dimensions(a, method=dimred, n_comps=n_comps, sqrt=False)
        y = pd.Categorical(a.obs['GBC'])
        Y = one_hot_from_labels(y)
    
    ##

    logger.info(f'Reading and formatting AFM, X and y, took total {t.stop()}')
    logger.info(f'Total cells and clones in the original QCed sample (perturb seq QC metrics): {ncells0}; {n_all_clones}.')
    logger.info(f'Total cells, clones and features submitted to classification: {ncells}; {n_clones_analyzed}, {X.shape[1]}.')


    ##


    # Here we go
    L = []
    trained_models = {}
    
    for i in range(Y.shape[1]):  
            
        t = Timer()
        t.start()
            
        comparison = f'{y.categories[i]}_vs_rest'
        y_ = Y[:,i]
        
        if y_.sum()>=10:

            logger.info(f'Starting comparison {comparison} ({i+1}/{Y.shape[1]})...')

            # Classification
            results = classification(
                X, y_, key=model, GS=True, GS_mode=GS_mode,
                score=score, n_combos=n_combos, cores_model=ncores, cores_GS=1, feature_names=variants
            )

            # Pack results up
            performance_dict = results['performance_dict']
            del results['performance_dict']

            # Performance
            performance_dict |= {
                'sample' : sample,
                'filtering' : filtering, 
                'dimred' : dimred,
                'min_cell_number' : min_cell_number,
                'ncells_clone' : y_.sum(),
                'ncells_sample' : ncells,
                'clone_prevalence' : y_.sum() / ncells,
                'n_clones_analyzed' : n_clones_analyzed,
                'n_features' : variants.size, 
                'model' : model,
                'tuning' : GS_mode,
                'score_for_tuning' : score,
                'comparison' : comparison
            }
            # Model + useful stuff 
            results |= {'cells':cells, 'variants':variants}

            # Store comparison performance_df and results pickle
            L.append(performance_dict)
            trained_models[comparison] = results

            logger.info(f'Finished {comparison} ({i+1}/{Y.shape[1]}): {t.stop()}')
        
        else:

            logger.info(f'Comparison {comparison} skipped: too low {y_.sum()} cells...')

    # Save results as csv
    df = pd.DataFrame(L)
    logger.info(df['f1'].describe())

    # Save all as a pickle
    path_results = os.path.join(
        path, 
        f'out_{sample}_{filtering_key}_{dimred}_{model}_{GS_mode}_{min_cell_number}.pickle'
    )
    with open(path_results, 'wb') as f:
        pickle.dump(
            {'performance_df':df, 'trained_models':trained_models},
            f
        )

    #-----------------------------------------------------------------#

    # Write final exec time
    logger.info(f'Execution was completed successfully in total {T.stop()} s.')

#######################################################################

# Run program
if __name__ == "__main__":
    main()













    

