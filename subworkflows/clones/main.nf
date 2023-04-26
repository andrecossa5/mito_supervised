// measter subworkflow

// Include here
nextflow.enable.dsl = 2
include { JOB } from "./modules/classification_clones.nf"

// 

//----------------------------------------------------------------------------//
// classification_clones subworkflow
//----------------------------------------------------------------------------//

workflow classification_clones {
    
    take:
        ch_samples 

    main:

        // Create options
        options = ch_samples
        .combine(params.filtering)
        .combine(params.GS_mode)
        .combine(params.dimred)
        .combine(params.model)
        .combine(params.min_cell_number)
        .filter{ !(it[1] != "pegasus" && it[3] != "no_dimred") }

        // Execute jobs
        JOB(options)

    emit:
        job_output = JOB.out.logs

}