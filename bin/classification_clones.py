#!/usr/bin/python

# Clones classification script

########################################################################

# Parsing CLI args 

# Libraries
import sys 
import argparse

# Create the parser
my_parser = argparse.ArgumentParser(
    prog='clones_classification',
    description=
        '''
        Systematically testing the ability of (filtered) MT-SNVs to distinguish ground truth clonal labels
        from lentiviral barcoding.
        '''
)

# Add arguments

# Path_main
my_parser.add_argument(
    '-p', 
    '--path_main', 
    type=str,
    default='..',
    help='The path to the main project directory. Default: .. .'
)

# Filter
my_parser.add_argument(
    '--sample', 
    type=str,
    default='MDA',
    help='Sample to use. Default: MDA.'
)

# Input_mode
my_parser.add_argument(
    '--input_mode', 
    type=str,
    default='more_stringent',
    help='Preprocessing version following the maegatk software. Default: more_stringent.'
)

# Filter
my_parser.add_argument(
    '--filtering', 
    type=str,
    default='ludwig2019',
    help='Method to filter MT-SNVs. Default: ludwig2019.'
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

# skip
my_parser.add_argument(
    '--skip', 
    action='store_true',
    help='Skip analysis. Default: False.'
)

# Parse arguments
args = my_parser.parse_args()

path_main = args.path_main
sample = args.sample
input_mode = args.input_mode
dimred = args.dimred
filtering = args.filtering if dimred == 'no_dimred' else 'pegasus'
model = args.model
ncombos = args.ncombos
ncores = args.ncores 
score = args.score
min_cell_number = args.min_cell_number
min_cov_treshold = args.min_cov_treshold
n_comps = args.n_comps
GS_mode = args.GS_mode

########################################################################

# Preparing run: import code, prepare directories, set logger
if not args.skip:

    # Code
    from MI_TO.utils import *
    from MI_TO.preprocessing import *
    from MI_TO.dimred import *
    from MI_TO.supervised import *

    #-----------------------------------------------------------------#

    # path_main = '/Users/IEO5505/Desktop/MI_TO/'
    path_data = path_main + 'data/'
    path_results = path_main + 'results_and_plots/supervised/clones_classification/'
    path_runs = path_main + 'runs/'

    #-----------------------------------------------------------------#

    # Set logger 
    logger = set_logger(path_runs, f'logs_{sample}_{input_mode}_{filtering}_{dimred}_{min_cell_number}_{min_cov_treshold}_{model}_{score}.txt')

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
        Execute classification: \n
        --sample {sample} 
        --input_mode {input_mode} 
        --filtering {filtering} 
        --dimred {dimred} 
        --model {model}
        --ncombos {ncombos} 
        --score {score} 
        --min_cell_number {min_cell_number} 
        --min_cov_treshold {min_cov_treshold}
        """
    )
    
    afm = read_one_sample(path_main, input_mode=input_mode, sample=sample)
    ncells0 = afm.shape[0]
    n_all_clones = len(afm.obs['GBC'].unique())
    blacklist = pd.read_csv(path_data + 'blacklist.csv', index_col=0)

    ##

    # Filter 'good quality' cells and variants
    if dimred == 'no_dimred':

        _, a = filter_cells_and_vars(
            afm,
            blacklist=blacklist,
            sample=sample,
            filtering=filtering, 
            min_cell_number=min_cell_number, 
            min_cov_treshold=min_cov_treshold, 
            nproc=ncores, 
            path_=path_results
        )

        # Extract X, y
        a = nans_as_zeros(a) # For sklearn APIs compatibility
        ncells = a.shape[0]
        n_clones_analyzed = len(a.obs['GBC'].unique())
        X = a.X
        feature_names = a.var_names
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
        X, _ = reduce_dimensions(a, method=dimred, n_comps=n_comps, sqrt=False)
        y = pd.Categorical(a.obs['GBC'])
        Y = one_hot_from_labels(y)
    
    ##

    logger.info(f'Reading and formatting AFM, X and y, took total {t.stop()}')
    logger.info(f'Total cells and clones in the original QCed sample (perturb seq QC metrics): {ncells0}; {n_all_clones}.')
    logger.info(f'Total cells, clones and features submitted to classification: {ncells}; {n_clones_analyzed}, {X.shape[1]}.')

    # Here we go
    L = []
    for i in range(Y.shape[1]):  
        
        t.start()
        comparison = f'{y.categories[i]}_vs_rest' 
        logger.info(f'Starting comparison {comparison}, {i+1}/{Y.shape[1]}...')

        y_ = Y[:,i]

        # Check numbers 

        if np.sum(y_) > min_cell_number:

            d = classification(X, y_, key=model, GS=True, GS_mode=GS_mode,
                score=score, n_combos=ncombos, cores_model=ncores, cores_GS=1)
            d |= {
                'sample' : sample,
                'filtering' : filtering, 
                'dimred' : dimred,
                'min_cell_number' : min_cell_number,
                'min_cov_treshold' : min_cov_treshold,
                'ncells' : ncells,
                'n_clones_analyzed' : n_clones_analyzed,
                'n_features' : X.shape[1],
                'model' : model,
                'score_for_tuning' : score,
                'comparison' : comparison
            }         
            L.append(d)
            logger.info(f'Comparison {comparison} finished: {t.stop()}')

        else:
            logger.info(f'Clone {y.categories[i]} does not reach {min_cell_number} cells. This should not happen here... {t.stop()}')

    df = pd.DataFrame(L)
    logger.info(df['f1'].describe())

    # Save results
    df.to_csv(path_results + f'{sample}_{input_mode}_{filtering}_{dimred}_{min_cell_number}_{min_cov_treshold}_{model}_{score}.csv')

    #-----------------------------------------------------------------#

    # Write final exec time
    logger.info(f'Execution was completed successfully in total {T.stop()} s.')

#######################################################################

# Run program
if __name__ == "__main__":
    if not args.skip:
        main()

#######################################################################














    

