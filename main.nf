// MI_TO pipeline
nextflow.enable.dsl = 2
include { classification_clones } from "./subworkflows/clones/main"
// include { classification_samples } from "./subworkflows/samples/main"

// Samples channel
ch_samples = Channel
    .fromPath("${params.path_data}/*", type:'dir') 
    .map{ it.getName() }

//

//----------------------------------------------------------------------------//
// mito_supervised entry points
//----------------------------------------------------------------------------//

//

workflow clones {

    classification_clones(ch_samples)
    classification_clones.out.job_output.view()

}

//

// workflow samples {
// 
//     classification_samples(ch_samples)
//     classification_samples.out.summary.view()
// 
// }

// Mock
workflow  {
    
    Channel.of(1,2,3,4) | view

}