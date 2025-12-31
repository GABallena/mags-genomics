# Portfolio-safe copy (Snakemake workflow)
# Identifiers generalized; no private data included.

import os
import glob
from pathlib import Path

# Get paths relative to Snakefile location
BASE_DIR = os.path.dirname(os.path.realpath(workflow.snakefile))
REFERENCE_DB = "markers/uniprot_sprot.fasta"  # Now in same directory
GOI_DIR = "markers/CARD"  # Now in same directory

# Rest of configuration
PROGRAM = "usearch"
THREADS = 8
SEARCH_IDENTITY = 0.95
OUTPUT_DIR = "results/markers"

# Validate input paths
if not os.path.exists(GOI_DIR):
    raise ValueError(f"CARD directory not found: {GOI_DIR}")
if not os.path.exists(REFERENCE_DB):
    raise ValueError(f"Reference database not found: {REFERENCE_DB}")

# Get CARD protein model files
CARD_MODELS = ["protein_fasta_protein_homolog_model.fasta"]

# Verify and get available models
goi_files = []
for model in CARD_MODELS:
    model_path = os.path.join(GOI_DIR, model)
    if os.path.exists(model_path):
        goi_files.append(os.path.splitext(model)[0])

if not goi_files:
    raise ValueError(f"No protein FASTA files found in {GOI_DIR}")

# Get trimmed reads (only if directory exists)
TRIMMED_DIR = "Project4/trimmed_reads"  # Changed to relative path
if not os.path.exists(TRIMMED_DIR):
    raise ValueError(f"Trimmed reads directory not found: {TRIMMED_DIR}")

trimmed_reads = [os.path.splitext(os.path.splitext(f)[0])[0] 
                 for f in os.listdir(TRIMMED_DIR) 
                 if f.endswith(".fastq.gz")]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

rule all:
    input:
        expand(f"{OUTPUT_DIR}/{{goi}}/{{sample}}/quantification.tsv", 
               goi=goi_files, sample=trimmed_reads)

rule decompress_reads:
    input:
        fastq_gz="trimmed_reads/{sample}.fastq.gz"  # Updated to relative path
    output:
        fasta="trimmed_reads/{sample}.fasta"  # Updated to relative path
    shell:
        """
        zcat {input.fastq_gz} | seqtk seq -A - > {output.fasta}
        """

rule generate_markers:
    input:
        goi=os.path.join(GOI_DIR, "{goi}.fasta"),
        reference=REFERENCE_DB
    output:
        markers=os.path.join(OUTPUT_DIR, "{goi}/markers.fasta"),
        map_file=os.path.join(OUTPUT_DIR, "{goi}/mapping.tsv"),
        plot_dir=directory(os.path.join(OUTPUT_DIR, "{goi}/plots"))
    params:
        script=os.path.join(BASE_DIR, "generate_markers.py")
    log:
        os.path.join(OUTPUT_DIR, "{goi}/generate_markers.log")
    shell:
        """
        set -e
        mkdir -p {output.plot_dir}
        python3 {params.script} \
            -i {input.goi} \
            -r {input.reference} \
            -o {output.markers} \
            -m {output.map_file} \
            --plot-dir {output.plot_dir} 2>&1 | tee {log}
        """

rule quantify_markers:
    input:
        markers=f"{OUTPUT_DIR}/{{goi}}/markers.fasta",
        wgs="Project4/trimmed_reads/{sample}.fasta"
    output:
        quantified=f"{OUTPUT_DIR}/{{goi}}/{{sample}}/quantification.tsv"
    params:
        script="quantify_markers.py",
        program=PROGRAM,
        threads=THREADS,
        identity=SEARCH_IDENTITY
    log:
        f"{OUTPUT_DIR}/{{goi}}/{{sample}}/quantify_markers.log"
    shell:
        """
        mkdir -p $(dirname {output.quantified}) && \
        python3 {params.script} \
            --markers {input.markers} \
            --wgs {input.wgs} \
            --output {output.quantified} \
            --program {params.program} \
            --threads {params.threads} \
            --identity {params.identity} > {log} 2>&1
        """
