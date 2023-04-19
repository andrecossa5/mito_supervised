#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// Paths
params.p = '/Users/IEO5505/Desktop/MI_TO/'
params.path_code = '/Users/IEO5505/Desktop/MI_TO/three_samples_analyses/scripts/'

//

// Create jobs options
process createOptions {

    output:
    path 'jobs.csv'

    script:
    """
    python $params.path_code/create_options.py > 'jobs.csv'
    """

}

// Run!
process runJobs {

    input:
    val x

    output:
    path 'out.csv'
    
    script:
    """
    python $params.path_code/classification_clones.py \
        --min_cov_treshold 50 \
        --ncombos 100 \
        --ncores ${task.cpus} \
        --p $params.p \
        --sample ${x[1]} \
        --input_mode ${x[2]} \
        --filtering ${x[3]} \
        --dimred ${x[4]} \
        --model ${x[5]} \
        --min_cell_number ${x[6]} > out.csv
    """

}

// Workflow
workflow { 

    jobs = createOptions().splitCsv(skip:1)
    runJobs(jobs) 
    view

}


