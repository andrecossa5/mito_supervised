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

        // Create job_channels
        ch_filtering = Channel.fromPath(params.path_filtering)
                        .map { file -> new groovy.json.JsonSlurper().parse(file).keySet() }
                        .flatMap()
        ch_input = ch_samples
            .combine(ch_filtering)
            .combine(params.GS_mode)
            .combine(params.dimred)
            .combine(params.model)
            .combine(params.min_cell_number)

        // Execute jobs
        JOB(ch_input)

    emit:
        job_output = JOB.out.output

}