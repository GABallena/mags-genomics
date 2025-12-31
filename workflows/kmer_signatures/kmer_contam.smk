# Portfolio-safe copy (Snakemake workflow)
# Identifiers generalized; no private data included.

# Include the other Snakefile
include: "what_distribution.smk"

# Configuration Parameters
CONTAM_DB = "databases/UniVec"
PHIX_DB = "databases/PhiX"
GENOME_1000_DB = "databases/1000_genomes"

# BUSCO Parameters
BUSCO_MODE = "genome"  # Mode for BUSCO (genome, proteins, transcriptome)
BUSCO_LINEAGE = "bacteria_odb10"  # Example lineage for BUSCO
BUSCO_THREADS = 4  # Number of CPU threads for BUSCO
BUSCO_EVALUE = "1e-05"  # E-value cutoff for BUSCO
BUSCO_OUT_NAME = "busco_validation"  # Output name for BUSCO runs


rule all:
    input:
        "results/cleaned_high_fidelity_spikes.fasta",
        "busco_validation_outputs/short_summary.txt"  # Final output after filtering and re-validating SCGs

# Step 1: Download UniVec, PhiX, and 1000 Genomes database
rule download_marker_genes:
    output:
        contam_db="databases/UniVec",
        phix_db="databases/PhiX",
        genome_1000_db="databases/1000_genomes.fasta"
    shell:
        """
        mkdir -p databases
        
        # Download UniVec contaminant sequences
        aria2c -x 16 -d databases https://ftp.ncbi.nlm.nih.gov/pub/UniVec/UniVec
        
        # Download PhiX contaminant sequences
        aria2c -x 16 -d databases ftp://igenome:<email>/phiX/illumina_phiX.fa.gz

        # Download 1000 Genomes Project data (VCF or FASTA format)
        aria2c -x 16 -d databases ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/release/20130502/ALL.chr*.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz
        """

# Step 2: Run BUSCO with all datasets (use all BUSCO lineages)
rule run_busco_all:
    input:
        "your_genome_or_assembly.fasta"
    output:
        directory("busco_outputs")
    conda:
        "env/busco_env.yaml"  # Use the BUSCO environment
    shell:
        """
        mkdir -p {output}
        
        # Run BUSCO with specified parameters
        busco -i {input} -m {BUSCO_MODE} -l {BUSCO_LINEAGE} -o {BUSCO_OUT_NAME} -c {BUSCO_THREADS} --evalue {BUSCO_EVALUE} --force --tar --restart --offline
        
        # Move results to the output folder
        mv {BUSCO_OUT_NAME}* {output}/
        """


# Step 3: Generate k-mers for SCGs using Jellyfish
rule generate_scg_kmers:
    input:
        expand("busco_outputs/{dataset}/short_summary.txt", dataset=["bacteria_odb10", "eukaryota_odb10", "others..."])
    output:
        "scg_marker_kmers.fa"
    conda:
        "env/jellyfish_env.yaml"  # Using the Jellyfish environment
    shell:
        """
        jellyfish count -m 31 -s 100M -C busco_outputs/*/single_copy_genes.fasta -o scg_marker_kmers.jf
        jellyfish dump scg_marker_kmers.jf > {output}
        """

# Step 4: Generate k-mers for UniVec
rule generate_contam_kmers:
    input:
        "databases/UniVec"
    output:
        "contaminant_kmers.fa"
    shell:
        """
        jellyfish count -m 31 -s 100M -C {input} -o contam_kmers.jf
        jellyfish dump contam_kmers.jf > {output}
        """

# Step 5: Generate k-mers for PhiX
rule generate_phix_kmers:
    input:
        "databases/PhiX"
    output:
        "phix_kmers.fa"
    shell:
        """
        jellyfish count -m 31 -s 100M -C {input} -o phix_kmers.jf
        jellyfish dump phix_kmers.jf > {output}
        """

# Step 6: Generate k-mers for 1000 Genomes Project
rule generate_genome_1000_kmers:
    input:
        "databases/1000_genomes.fasta"
    output:
        "genome_1000_kmers.fa"
    shell:
        """
        jellyfish count -m 31 -s 100M -C {input} -o genome_1000_kmers.jf
        jellyfish dump genome_1000_kmers.jf > {output}
        """

# Step 7: Map SCG k-mers to raw reads using KMA
rule map_scg_kmers:
    input:
        reads="raw_reads.fastq",
        kmers="scg_marker_kmers.fa"
    output:
        "results/scg_kmer_stats.txt"
    shell:
        """
        kma -i {input.reads} -t_db {input.kmers} -o results/kma_scg_output
        mv results/kma_scg_output.res {output}
        """

# Step 8: Map UniVec k-mers to raw reads using KMA
rule map_contam_kmers:
    input:
        reads="raw_reads.fastq",
        kmers="contaminant_kmers.fa"
    output:
        "results/contam_kmer_stats.txt"
    shell:
        """
        kma -i {input.reads} -t_db {input.kmers} -o results/kma_contam_output
        mv results/kma_contam_output.res {output}
        """

# Step 9: Map PhiX k-mers to raw reads using KMA
rule map_phix_kmers:
    input:
        reads="raw_reads.fastq",
        kmers="phix_kmers.fa"
    output:
        "results/phix_kmer_stats.txt"
    shell:
        """
        kma -i {input.reads} -t_db {input.kmers} -o results/kma_phix_output
        mv results/kma_phix_output.res {output}
        """

# Step 10: Map 1000 Genomes k-mers to raw reads using KMA
rule map_genome_1000_kmers:
    input:
        reads="raw_reads.fastq",
        kmers="genome_1000_kmers.fa"
    output:
        "results/genome_1000_kmer_stats.txt"
    shell:
        """
        kma -i {input.reads} -t_db {input.kmers} -o results/kma_genome_1000_output
        mv results/kma_genome_1000_output.res {output}
        """

# Step 11: Normalize contaminants by k-mer using wavelets
rule normalize_contam_by_kmer:
    input:
        "results/contam_kmer_stats.txt"
    output:
        "results/wavelet_normalized_contam.txt"
    script:
        "scripts/normalize_contam_wavelet.py"

# Step 12: Normalize SCGs by k-mer using wavelets
rule normalize_scgs_by_kmer:
    input:
        "results/scg_kmer_stats.txt"
    output:
        "results/wavelet_normalized_scgs.txt"
    script:
        "scripts/normalize_scgs_wavelet.py"

# Segway 1: A "few" tests
rule check_distribution:
    input:
        "kmer_counts.txt"
    output:
        rule check_distribution:
    input:
        "kmer_counts.txt"
    output:
        "distribution_check.txt"
    script:
        "scripts/normality_test.py"

rule fit_alternate_distributions:
    input:
        "distribution_check.txt"
    output:
        "fitted_distributions.txt"
    script:
        "scripts/fit_distributions.py"

rule transform_data_if_needed:
    input:
        "fitted_distributions.txt"
    output:
        "transformed_data.txt"  
    script:
        "scripts/goodness_of_fit.py"  

rule goodness_of_fit_parametric:
    input:
        "fitted_distributions.txt"
    output:
        "goodness_of_fit.txt" 
    script:
        "scripts/goodness_of_fit.py"  

rule non_parametric_workflow:
    input:
        "fitted_distributions.txt"
    output:
        "non_parametric_results.txt"
    script:
        "scripts/non_parametric_analysis.py"

rule goodness_of_fit_nonparametric:
    input:
        "kmer_counts.txt"
    output:
        "goodness_of_fit.txt"  
    script:
        "scripts/goodness_of_fit_nonparametric.py"

rule best_fit:
    input:
        "kmer_counts.txt"
    output:
        "fit_results.tsv" 
    script:
        "best_fit.py"
    conda: "env/distribution.yaml"
    script:
        "what_distribution.smk"


# Best fit and alpha
rule best_fit_plus_alpha:
    input:
        "fit_results.tsv",
        "goodness_of_fit.txt"
    output:
        "best_fit_and_alpha.txt"  # Define where to store the best fit model and optimal alpha
    script:
        "scripts/bestfit_and_alpha.py"  # Path to your script



# Step 13: Unified workflow for parametric and non-parametric tests
rule unified_statistical_workflow:
    input:
        contam="results/wavelet_normalized_contam.txt",
        scgs="results/wavelet_normalized_scgs.txt",
        bestfit_alpha="results/bestfit_alpha.json"
    output:
        "results/statistical_test_results.txt",
        ambiguous_sequences="results/ambiguous_sequences.txt"
    script:
        "scripts/compare_wavelets_stats.py"



# Step 14: Filter ambiguous sequences and keep only high-quality reads
rule filter_ambiguous_sequences:
    input:
        assembled="results/spikes_assembled.fasta",
        ambiguous="results/ambiguous_sequences.txt"
    output:
        "results/cleaned_high_fidelity_spikes.fasta"
    shell:
        """
        # Remove ambiguous sequences from assembled spikes
        grep -vFf {input.ambiguous} {input.assembled} > {output}
        """


# Step 15: Re-run BUSCO to validate SCG retention after filtering
rule rerun_busco:
    input:
        "results/cleaned_high_fidelity_spikes.fasta"
    output:
        directory("busco_validation_outputs")
    conda:
        "env/busco_env.yaml"  # Use the BUSCO environment
    shell:
        """
        mkdir -p {output}
        
        # Run BUSCO with specified parameters
        busco -i {input} -m {BUSCO_MODE} -l {BUSCO_LINEAGE} -o {BUSCO_OUT_NAME} -c {BUSCO_THREADS} --evalue {BUSCO_EVALUE} --force --tar --restart --offline
        
        # Move results to the output folder
        mv {BUSCO_OUT_NAME}* {output}/
        """
