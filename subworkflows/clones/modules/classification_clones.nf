// JOB module

nextflow.enable.dsl = 2

//

process JOB {

    tag "${sample}: ${filtering_key}, ${dimred}, ${model}, ${GS_mode}, ${min_cell_number}"
    publishDir "${params.outdir}/", mode: 'copy'

    input:
    tuple val(sample), 
        val(filtering_key),
        val(GS_mode),
        val(dimred),
        val(model),
        val(min_cell_number)

    output:
    path "out_${sample}_${filtering_key}_${dimred}_${model}_${GS_mode}_${min_cell_number}.pickle", emit: output
    path "log_${sample}_${filtering_key}_${dimred}_${model}_${GS_mode}_${min_cell_number}.txt", emit: logs
    
    script:
    """
    python ${baseDir}/bin/classification_clones.py \
    --path_data ${params.path_data} \
    --path_priors ${params.path_priors} \
    --ncombos ${params.n_combos} \
    --ncores ${task.cpus} \
    --sample ${sample} \
    --score ${params.score} \
    --path_filtering ${params.path_filtering} \
    --filtering_key ${filtering_key} \
    --GS_mode ${GS_mode} \
    --dimred ${dimred} \
    --model ${model} \
    --min_cell_number ${min_cell_number}
    """

    stub:
    """
    touch "out_${sample}_${filtering_key}_${dimred}_${model}_${GS_mode}_${min_cell_number}.pickle"
    touch "log_${sample}_${filtering_key}_${dimred}_${model}_${GS_mode}_${min_cell_number}.txt"
    """

}
