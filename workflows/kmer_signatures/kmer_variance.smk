# Portfolio-safe copy (Snakemake workflow)
# Identifiers generalized; no private data included.

rule all:
    input:
        "results/kmer_matrix.csv",
        "results/variance_per_scg.txt",
        "results/output_bowtie.sam",
        "results/output_bwa.sam",
        "results/output_minimap.sam",
        "results/output_kma_res.txt"

# K-mer steps remain as in the previous Snakemake file
# ---------------------------------
rule generate_kmers_for_scgs:
    input:
        scg_fasta="data/scg_genes.fasta"
    output:
        kmers="results/scg_kmers.fa"
    shell:
        """
        jellyfish count -m 31 -s 100M -C {input.scg_fasta} -o results/scg_kmers.jf
        jellyfish dump results/scg_kmers.jf > {output.kmers}
        """

rule construct_kmer_matrix:
    input:
        kmers="results/scg_kmers.fa"
    output:
        matrix="results/kmer_matrix.csv"
    script:
        "scripts/construct_kmer_matrix.py"

rule check_kmer_variance:
    input:
        matrix="results/kmer_matrix.csv"
    output:
        "results/variance_per_scg.txt"
    script:
        "scripts/check_kmer_variance.py"

# Alignment steps with 95% sequence identity
# ---------------------------------
rule bowtie2_align:
    input:
        reference="data/reference.fasta",
        reads="data/reads.fastq"
    output:
        "results/output_bowtie.sam"
    shell:
        """
        bowtie2 -x {input.reference} -U {input.reads} -N 1 --mp 2,2 --score-min L,0.95,0.1 -S {output}
        """

rule bwa_mem_align:
    input:
        reference="data/reference.fasta",
        reads="data/reads.fastq"
    output:
        "results/output_bwa.sam"
    shell:
        """
        bwa mem -B 4 -O 6 -E 1 {input.reference} {input.reads} > {output}
        """

rule minimap2_align:
    input:
        reference="data/reference.fasta",
        reads="data/reads.fastq"
    output:
        "results/output_minimap.sam"
    shell:
        """
        minimap2 -a -x map-ont --secondary=no -p 0.95 {input.reference} {input.reads} > {output}
        """

rule kma_validate:
    input:
        reference="data/reference.fasta",
        reads="data/reads.fastq"
    output:
        "results/output_kma_res.txt"
    shell:
        """
        kma -i {input.reads} -o results/output_kma -t_db {input.reference} -mp 95 -mem_mode -1t1
        mv results/output_kma.res {output}
        """
