# Usage (high-level)

Typical pattern:
1. Prepare input FASTA/FASTQ (or contigs).
2. Construct k-mer matrices (`construct_kmer_matrix.py`).
3. Evaluate variance/diversity across samples (`check_kmer_variance.py`, workflows).
4. Optionally run contamination-oriented checks (`kmer_contam.smk`).
5. Generate marker sets (`generate_markers.py`, `marker_generation.smk`).
