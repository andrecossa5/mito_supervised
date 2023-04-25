// JOB module

nextflow.enable.dsl = 2

//

process JOB {

    tag "${sample}_${filtering}_${dimred}_${model}_${GS_mode}_${min_cell_number}"
    publishDir "${params.outdir}/", mode: 'copy'

    input:
    tuple val(sample), 
        val(filtering),
        val(GS_mode),
        val(dimred),
        val(model),
        val(min_cell_number)

    output:
    path "out_${sample}_${filtering}_${dimred}_${model}_${GS_mode}_${min_cell_number}.csv", emit: output
    path "log_${sample}_${filtering}_${dimred}_${model}_${GS_mode}_${min_cell_number}.txt", emit: logs
    
    script:
    """
    python ${baseDir}/bin/classification_clones.py \
        --min_cov_treshold ${params.min_cov_treshold} \
        --ncombos ${params.n_combos} \
        --ncores ${task.cpus} \
        --p ${params.path_data} \
        --sample ${sample} \
        --filtering ${filtering} \
        --GS_mode ${GS_mode} \
        --dimred ${dimred} \
        --model ${model} \
        --min_cell_number ${min_cell_number}
    """

    stub:
    """
    touch "out_${sample}_${filtering}_${dimred}_${model}_${GS_mode}_${min_cell_number}.csv"
    touch "log_${sample}_${filtering}_${dimred}_${model}_${GS_mode}_${min_cell_number}.txt"
    """


}
