#!/usr/bin/env python3
# Portfolio-safe copy: extract spike-in/marker sequences to FASTA
# Identifiers generalized; no private data included.

import numpy as np
from Bio import SeqIO

# Load normalized SCG data
normalized_scgs = np.loadtxt('results/wavelet_normalized_scgs.txt')

# Set threshold for identifying spikes (e.g., Z-score > 3)
threshold = 3

# Identify indices where spikes occur
spike_indices = np.where(normalized_scgs > threshold)[0]

# Load raw reads (FASTA/FASTQ format) and extract the corresponding spikes
input_reads = "raw_reads.fastq"  # You can change this to FASTA if needed
output_fasta = "results/spikes_of_interest.fasta"

# Write spikes to the output FASTA file
with open(output_fasta, "w") as output_handle:
    for i, record in enumerate(SeqIO.parse(input_reads, "fastq")):
        if i in spike_indices:
            SeqIO.write(record, output_handle, "fasta")

print(f"Extracted {len(spike_indices)} spikes and saved to {output_fasta}")
