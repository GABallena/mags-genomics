# Portfolio-safe copy (Snakemake workflow)
# Identifiers generalized; no private data included.

# Define the directory containing cleaned reads
CLEANED_READS_DIR = "cleaned_reads"

# Define the directory containing Kraken taxonomic profiles
KRAKEN_OUTPUT_DIR = "Kraken/kraken_output"

# Automatically generate site names if they follow a pattern
SITES = [f"SSR{i}" for i in range(1, 101)]  # Adjust the range as needed

# Final rule to tie everything together
rule all:
    input:
        expand("entropy_results/{site}_entropy_comparison.txt", site=SITES)

# Rule to generate k-mer profiles using Jellyfish for paired-end reads
rule jellyfish_count:
    conda: "env/jellyfish_env.yaml"  # Specify the YAML file for Jellyfish environment
    input:
        r1 = CLEANED_READS_DIR + "/{site}_R1_paired.fastq.gz",
        r2 = CLEANED_READS_DIR + "/{site}_R2_paired.fastq.gz"
    output:
        "entropy_results/{site}.kmer_counts"
    params:
        mer_size = 31,
        count_cutoff = 2
    shell:
        """
        jellyfish count -m {params.mer_size} -s 100M -C {input.r1} {input.r2} -o {output}.jf
        jellyfish dump {output}.jf | awk '{{if ($2 >= {params.count_cutoff}) print $0}}' > {output}
        """
        
# Rule to create MASH sketches from k-mer profiles
rule mash_sketch:
    conda: "env/mash_env.yaml"  # Specify the YAML file for MASH environment
    input:
        "entropy_results/{site}.kmer_counts"
    output:
        "entropy_results/{site}.mash_sketch.msh"
    params:
        sketch_size = 10000000
    shell:
        """
        mash sketch -k 31 -s {params.sketch_size} -o {output} {input}
        """

# Rule to calculate Shannon entropy for k-mers
rule shannon_entropy_kmers:
    input:
        "entropy_results/{site}.kmer_counts"
    output:
        "entropy_results/{site}_shannon_entropy_kmers.txt"
    params:
        sample_size = 10000
    shell:
        """
        awk '{{print $1}}' {input} | shuf -n {params.sample_size} | \
        jellyfish histo /dev/stdin | \
        awk '{{entropy -= $1 * log($1)}} END {{print entropy}}' > {output}
        """
        
# Rule to calculate Shannon entropy for taxonomic profiles using Kraken2 reports
rule shannon_entropy_taxa:
    input:
        report = KRAKEN_OUTPUT_DIR + "/{site}.report"
    output:
        "entropy_results/{site}_shannon_entropy_taxa.txt"
    shell:
        """
        awk '{{counts[$1] += $2}} END {{for (taxon in counts) {{ p = counts[taxon] / SUM; entropy -= p * log(p) }}; print entropy}}' {input.report} > {output}
        """

# Rule to compare Shannon entropy metrics
rule compare_entropy_metrics:
    input:
        kmer_entropy = "entropy_results/{site}_shannon_entropy_kmers.txt",
        taxa_entropy = "entropy_results/{site}_shannon_entropy_taxa.txt"
    output:
        "entropy_results/{site}_entropy_comparison.txt"
    shell:
        """
        echo "K-mer Entropy: $(cat {input.kmer_entropy})" > {output}
        echo "Taxonomic Entropy: $(cat {input.taxa_entropy})" >> {output}
        """
