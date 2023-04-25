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
        options = ch_samples
        .combine(params.filtering)
        .combine(params.GS_mode)
        .combine(params.dimred)
        .combine(params.model)
        .combine(params.min_cell_number)
        JOB(options)

    emit:
        job_output = JOB.out.output

}