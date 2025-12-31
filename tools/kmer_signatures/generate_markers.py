#!/usr/bin/env python3
# Portfolio-safe copy: marker generation from sequence features
# Identifiers generalized; no private data included.

#!/usr/bin/env python3

# Standard library imports
import sys
import os
import time
import argparse
from collections import defaultdict
from tempfile import NamedTemporaryFile
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import subprocess
import multiprocessing as mp

# Third-party imports
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio import motifs
from Levenshtein import distance
import psutil
from tqdm import tqdm
import joblib
import numba
from numba import cuda, jit
import cupy as cp
import torch
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import MiniBatchKMeans
from datasketch import MinHash, MinHashLSH

# Local imports
from typing import Dict, List, Tuple, Optional, Union
from visualize import (plot_distance_matrix, plot_cluster_dendrogram,
                      plot_motif_logo, plot_marker_statistics, plot_cluster_sizes)
import logging

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('marker_generation_detailed.log')
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process()
    mem_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    logger.debug(f"Current memory usage: {mem_usage:.2f} MB")

def log_performance_decorator(func):
    """Decorator to log function performance metrics."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        logger.debug(f"Starting {func.__name__}")
        log_memory_usage()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        duration = end_time - start_time
        memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
        
        logger.debug(f"Completed {func.__name__}")
        logger.debug(f"Duration: {duration:.2f} seconds")
        logger.debug(f"Memory delta: {memory_delta:.2f} MB")
        return result
    return wrapper

@log_performance_decorator
def load_sequences(filepath: str) -> Dict[str, str]:
    """
    Load biological sequences from a FASTA format file, handling aligned sequences.
    
    Parameters:
    -----------
    filepath : str
        Path to the FASTA file containing sequences
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping sequence IDs to their sequence strings
        
    Notes:
    ------
    - Removes alignment gaps ('-' characters) from sequences
    - Validates sequence format and content
    - Logs sequence loading progress
    """
    logger.info(f"Loading sequences from: {filepath}")
    sequences = {}
    total_size = os.path.getsize(filepath)
    logger.debug(f"File size: {total_size/1024:.2f} KB")
    
    try:
        records = list(SeqIO.parse(filepath, "fasta"))
        logger.debug(f"Found {len(records)} sequences in file")
        
        for record in tqdm(records, desc="Loading sequences"):
            # Detailed sequence validation
            seq = str(record.seq).replace('-', '')
            logger.debug(f"Processing sequence {record.id}:")
            logger.debug(f"  Original length: {len(record.seq)}")
            logger.debug(f"  Cleaned length: {len(seq)}")
            logger.debug(f"  GC content: {(seq.count('G') + seq.count('C'))/len(seq)*100:.2f}%")
            
            if not seq:
                logger.warning(f"Empty sequence found for ID: {record.id}")
                continue
            
            sequences[record.id] = seq
            
        logger.info(f"Successfully loaded {len(sequences)} sequences")
        log_memory_usage()
        return sequences
    
    except Exception as e:
        logger.error(f"Error loading sequences from {filepath}: {str(e)}")
        raise

@lru_cache(maxsize=1024)
def cached_distance(seq1: str, seq2: str) -> float:
    """Cached version of Levenshtein distance calculation."""
    return distance(seq1, seq2)

def batch_distance_calculation(args):
    """Process a batch of distance calculations."""
    i, goi_seq, ref_seqs, threshold = args
    distances = np.zeros(len(ref_seqs))
    for j, ref_seq in enumerate(ref_seqs):
        dist = cached_distance(goi_seq, ref_seq)
        norm_dist = dist / max(len(goi_seq), len(ref_seq))
        distances[j] = norm_dist if norm_dist <= threshold else 1.0
    return i, distances

@log_performance_decorator
def calculate_distance_matrix(
    goi_seqs: Dict[str, str],
    ref_seqs: Dict[str, str],
    threshold: float = 0.8,
    batch_size: int = 100
) -> np.ndarray:
    """Optimized parallel distance matrix calculation."""
    logger.info("Calculating distance matrix with parallel processing...")
    
    goi_ids = list(goi_seqs.keys())
    ref_ids = list(ref_seqs.keys())
    matrix = np.zeros((len(goi_ids), len(ref_ids)))
    
    # Prepare batches for parallel processing
    batches = [(i, goi_seqs[goi_id], list(ref_seqs.values()), threshold)
               for i, goi_id in enumerate(goi_ids)]
    
    # Use process pool for parallel computation
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for i, distances in executor.map(batch_distance_calculation, batches):
            matrix[i] = distances
            
    return matrix

@log_performance_decorator
def find_unique_regions(goi_seq: str, ref_seqs: Dict[str, str], min_length: int = 8, max_length: int = 50) -> List[Tuple[int, int, str]]:
    """
    Find unique regions in a GOI sequence, considering alignment.
    
    Parameters:
    -----------
    goi_seq : str
        Gene of interest sequence
    ref_seqs : Dict[str, str]
        Dictionary of reference sequences
    min_length : int
        Minimum length of unique region (default: 8)
    max_length : int
        Maximum length of unique region (default: 50)
        
    Returns:
    --------
    List[Tuple[int, int, str]]
        List of tuples containing start index, end index, and unique region sequence
        
    Notes:
    ------
    - Removes alignment gaps from sequences
    - Identifies regions in GOI sequence not present in any reference sequence
    - Logs progress and results
    """
    logger.info("Finding unique regions in GOI sequence...")
    unique_regions = []
    clean_goi = goi_seq.replace('-', '')
    clean_refs = {k: v.replace('-', '') for k, v in ref_seqs.items()}
    
    for i in range(len(clean_goi) - min_length):
        for j in range(i + min_length, min(i + max_length, len(clean_goi))):
            candidate = clean_goi[i:j]
            if not any(candidate in ref_seq for ref_seq in clean_refs.values()):
                unique_regions.append((i, j, candidate))
    
    logger.info(f"Found {len(unique_regions)} unique regions")
    return unique_regions

@numba.jit(nopython=True, parallel=True)
def fast_distance_calculation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Optimized Levenshtein distance calculation using Numba."""
    len1, len2 = len(arr1), len(arr2)
    if len1 < len2:
        return fast_distance_calculation(arr2, arr1)
    if len2 == 0:
        return float(len1)

    # Initialize arrays with float64 type
    previous_row = np.arange(len2 + 1, dtype=np.float64)
    current_row = np.zeros(len2 + 1, dtype=np.float64)
    
    for i in numba.prange(len1):
        current_row[0] = float(i + 1)
        for j in range(len2):
            insertions = previous_row[j + 1] + 1.0
            deletions = current_row[j] + 1.0
            substitutions = previous_row[j] + (1.0 if arr1[i] != arr2[j] else 0.0)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row[:] = current_row[:]  # Copy values instead of reference
    
    return float(current_row[-1])

def parallel_distance_chunk(args):
    """Process a chunk of sequences for distance calculation."""
    chunk_goi, ref_seqs, threshold = args
    chunk_matrix = np.zeros((len(chunk_goi), len(ref_seqs)))
    
    for i, (goi_id, goi_seq) in enumerate(chunk_goi.items()):
        # Convert string to numeric array for Numba
        goi_array = np.array([ord(c) for c in goi_seq], dtype=np.int64)
        for j, (ref_id, ref_seq) in enumerate(ref_seqs.items()):
            ref_array = np.array([ord(c) for c in ref_seq], dtype=np.int64)
            dist = fast_distance_calculation(goi_array, ref_array)
            norm_dist = dist / max(len(goi_seq), len(ref_seq))
            chunk_matrix[i, j] = norm_dist if norm_dist <= threshold else 1.0
    
    return chunk_matrix

def try_gpu_clustering(goi_seqs, ref_seqs, threshold):
    """Attempt GPU-accelerated clustering if CUDA is available."""
    try:
        # Convert sequences to numerical arrays for GPU processing
        goi_arrays = np.array([list(map(ord, seq)) for seq in goi_seqs.values()])
        ref_arrays = np.array([list(map(ord, seq)) for seq in ref_seqs.values()])
        
        # Move data to GPU
        goi_gpu = cp.array(goi_arrays)
        ref_gpu = cp.array(ref_arrays)
        
        # Calculate distances on GPU
        distances = cp.zeros((len(goi_seqs), len(ref_seqs)))
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (distances.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (distances.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )
        
        gpu_distance_kernel[blocks_per_grid, threads_per_block](
            goi_gpu, ref_gpu, distances, threshold)
        
        return cp.asnumpy(distances)
    except:
        return None

@cuda.jit
def gpu_distance_kernel(goi_seqs, ref_seqs, distances, threshold):
    """CUDA kernel for parallel distance calculation."""
    i, j = cuda.grid(2)
    if i < distances.shape[0] and j < distances.shape[1]:
        dist = 0
        max_len = max(len(goi_seqs[i]), len(ref_seqs[j]))
        for k in range(min(len(goi_seqs[i]), len(ref_seqs[j]))):
            if goi_seqs[i][k] != ref_seqs[j][k]:
                dist += 1
        norm_dist = dist / max_len
        distances[i, j] = norm_dist if norm_dist <= threshold else 1.0

def get_optimal_compute_device(data_size: int) -> str:
    """Determine the optimal compute device based on data size and system capabilities."""
    logger.info("Determining optimal compute device...")
    
    # Check GPU availability and memory
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        logger.debug(f"GPU memory available: {gpu_memory/1e9:.2f} GB")
    
    # Check CPU resources
    cpu_count = mp.cpu_count()
    system_memory = psutil.virtual_memory().total
    logger.debug(f"CPU cores: {cpu_count}, System memory: {system_memory/1e9:.2f} GB")
    
    # Decision logic
    if gpu_available and data_size > 1e6:  # Large dataset
        return "gpu"
    elif cpu_count >= 8 and system_memory > 16e9:  # Good CPU resources
        return "multicore"
    else:
        return "basic"

class AdaptiveClusterer:
    """Adaptive clustering with hybrid approach."""
    
    def __init__(self, n_permutations=128, threshold=0.8):
        self.n_permutations = n_permutations
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold)
        
    def _sequence_to_minhash(self, sequence: str) -> MinHash:
        """Convert sequence to MinHash representation."""
        mh = MinHash(num_perm=self.n_permutations)
        # Use k-mers for better sensitivity
        k = 5
        for i in range(len(sequence) - k + 1):
            mh.update(sequence[i:i+k].encode('utf-8'))
        return mh
    
    def cluster_sequences(self, goi_seqs: Dict[str, str], ref_seqs: Dict[str, str]) -> Dict[str, List[str]]:
        """Perform hybrid clustering using MinHash LSH and fine-tuning."""
        logger.info("Starting hybrid clustering...")
        
        # Phase 1: Fast approximate clustering with MinHash
        sequence_hashes = {}
        for seq_id, seq in {**goi_seqs, **ref_seqs}.items():
            mh = self._sequence_to_minhash(seq)
            sequence_hashes[seq_id] = mh
            self.lsh.insert(seq_id, mh)
        
        # Phase 2: Form initial clusters
        initial_clusters = defaultdict(list)
        for goi_id in goi_seqs:
            neighbors = self.lsh.query(sequence_hashes[goi_id])
            if neighbors:
                ref_neighbors = [n for n in neighbors if n in ref_seqs]
                if ref_neighbors:
                    initial_clusters[ref_neighbors[0]].append(goi_id)
                else:
                    initial_clusters[goi_id].append(goi_id)
        
        # Phase 3: Fine-tune clusters using Levenshtein distance
        final_clusters = defaultdict(list)
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            for ref_id, members in initial_clusters.items():
                futures.append(executor.submit(
                    self._refine_cluster,
                    ref_id,
                    members,
                    goi_seqs,
                    ref_seqs
                ))
            
            for future in futures:
                ref_id, members = future.result()
                final_clusters[ref_id].extend(members)
        
        return final_clusters
    
    def _refine_cluster(self, ref_id: str, members: List[str],
                       goi_seqs: Dict[str, str], ref_seqs: Dict[str, str]) -> Tuple[str, List[str]]:
        """Refine cluster using Levenshtein distance."""
        refined_members = []
        ref_seq = ref_seqs.get(ref_id, goi_seqs[ref_id])
        
        for member_id in members:
            dist = fast_distance_calculation(goi_seqs[member_id], ref_seq)
            if dist <= self.threshold:
                refined_members.append(member_id)
        
        return ref_id, refined_members

class EnhancedMotifScorer:
    """Enhanced motif scoring with multiple methods."""
    
    def __init__(self):
        self.hmm_scores = {}
        self.pwm_scores = {}
        
    def score_sequence(self, sequence: str, motif_instances: List[motifs.Motif]) -> float:
        """Score sequence using multiple methods."""
        # PWM scoring
        pwm_score = self._calculate_pwm_score(sequence, motif_instances)
        
        # HMM scoring
        hmm_score = self._calculate_hmm_score(sequence, motif_instances)
        
        # Combine scores (weighted average)
        combined_score = 0.6 * pwm_score + 0.4 * hmm_score
        return combined_score
    
    def _calculate_pwm_score(self, sequence: str, motif_instances: List[motifs.Motif]) -> float:
        """Calculate PWM-based score."""
        scores = []
        for motif in motif_instances:
            pwm = motif.counts.normalize(pseudocounts=0.5)
            scores.extend([pwm.calculate(sequence[i:i+len(motif)])
                         for i in range(len(sequence) - len(motif) + 1)
                         if i + len(motif) <= len(sequence)])
        return max(scores) if scores else 0
    
    def _calculate_hmm_score(self, sequence: str, motif_instances: List[motifs.Motif]) -> float:
        """Calculate HMM-based score."""
        # Build HMM from motif instances
        hmm_scores = []
        for motif in motif_instances:
            hmm_model = self._build_hmm_from_motif(motif)
            score = hmmsearch(hmm_model, sequence)
            hmm_scores.append(score)
        return max(hmm_scores) if hmm_scores else 0
    
    def _build_hmm_from_motif(self, motif: motifs.Motif):
        """Build HMM model from motif."""
        # Implementation depends on specific HMM library
        pass

class PerformanceMonitor:
    """Monitor and optimize pipeline performance."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        self.metrics[operation_name].append({
            'start': time.time(),
            'memory_start': psutil.Process().memory_info().rss
        })
    
    def end_operation(self, operation_name: str):
        """End timing an operation and record metrics."""
        if operation_name in self.metrics:
            op_data = self.metrics[operation_name][-1]
            op_data['end'] = time.time()
            op_data['memory_end'] = psutil.Process().memory_info().rss
            op_data['duration'] = op_data['end'] - op_data['start']
            op_data['memory_delta'] = op_data['memory_end'] - op_data['memory_start']
    
    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        stats = {}
        for op_name, measurements in self.metrics.items():
            stats[op_name] = {
                'avg_duration': np.mean([m['duration'] for m in measurements]),
                'max_memory': max([m['memory_delta'] for m in measurements]),
                'call_count': len(measurements)
            }
        return stats

# Update cluster_sequences to use new adaptive approach
@log_performance_decorator
def cluster_sequences(goi_seqs: Dict[str, str], ref_seqs: Dict[str, str], 
                     distance_threshold: float = 0.2,
                     chunk_size: int = 1000) -> Dict[str, List[str]]:
    """
    Highly optimized parallel sequence clustering with GPU acceleration.
    """
    logger.info("Starting optimized sequence clustering...")
    clusters = defaultdict(list)
    
    # Try GPU clustering first
    distance_matrix = try_gpu_clustering(goi_seqs, ref_seqs, distance_threshold)
    
    if distance_matrix is None:
        logger.info("Falling back to CPU parallel processing...")
        # Split sequences into chunks for parallel processing
        goi_items = list(goi_seqs.items())
        chunks = [
            dict(goi_items[i:i + chunk_size])
            for i in range(0, len(goi_items), chunk_size)
        ]
        
        # Calculate distances in parallel using process pool
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            chunk_args = [(chunk, ref_seqs, distance_threshold) for chunk in chunks]
            chunk_results = list(executor.map(parallel_distance_chunk, chunk_args))
        
        # Combine chunk results
        distance_matrix = np.vstack(chunk_results)
    
    # Use numpy operations for faster clustering
    goi_ids = list(goi_seqs.keys())
    ref_ids = list(ref_seqs.keys())
    
    # Vectorized operations for cluster assignment
    min_distances = np.min(distance_matrix, axis=1)
    best_matches = np.argmin(distance_matrix, axis=1)
    
    # Fast cluster assignment using numpy boolean indexing
    cluster_mask = min_distances <= distance_threshold
    
    # Vectorized cluster assignment
    for i, (is_clustered, best_match) in enumerate(zip(cluster_mask, best_matches)):
        if is_clustered:
            clusters[ref_ids[best_match]].append(goi_ids[i])
        else:
            clusters[goi_ids[i]].append(goi_ids[i])
    
    logger.info(f"Clustering complete: {len(clusters)} clusters formed")
    return clusters

def run_alignment(sequences: Dict[str, str], method: str) -> str:
    """
    Run alignment using specified method.
    
    Parameters:
    -----------
    sequences : Dict[str, str]
        Dictionary of sequences to align
    method : str
        Alignment method ('mafft', 'tcoffee', or 'clustalw')
        
    Returns:
    --------
    str
        Path to the output alignment file
        
    Notes:
    ------
    - Writes sequences to a temporary file
    - Runs alignment using specified method
    - Logs alignment progress and results
    """
    logger.info(f"Running alignment using method: {method}")
    with NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as temp_in, \
         NamedTemporaryFile(suffix='.aln', delete=False) as temp_out:
        
        for seq_id, seq in sequences.items():
            temp_in.write(f">{seq_id}\n{seq}\n")
        temp_in.flush()
        
        if method == 'mafft':
            cmd = f"mafft --auto {temp_in.name} > {temp_out.name}"
        elif method == 'tcoffee':
            cmd = f"t_coffee {temp_in.name} -output fasta_aln -outfile {temp_out.name}"
        elif method == 'clustalw':
            cmd = f"clustalw -INFILE={temp_in.name} -OUTFILE={temp_out.name} -OUTPUT=FASTA"
        
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Alignment complete: {temp_out.name}")
        return temp_out.name

def run_mcoffee(alignment_files: List[str]) -> str:
    """
    Run M-Coffee meta-alignment.
    
    Parameters:
    -----------
    alignment_files : List[str]
        List of paths to alignment files
        
    Returns:
    --------
    str
        Path to the output meta-alignment file
        
    Notes:
    ------
    - Runs M-Coffee meta-alignment on provided alignment files
    - Logs meta-alignment progress and results
    """
    logger.info("Running M-Coffee meta-alignment...")
    with NamedTemporaryFile(suffix='.aln', delete=False) as temp_out:
        cmd = f"m_coffee -in {' '.join(alignment_files)} -output fasta_aln -outfile {temp_out.name}"
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"M-Coffee meta-alignment complete: {temp_out.name}")
        return temp_out.name

def refine_clusters(clusters: Dict[str, List[str]], goi_seqs: Dict[str, str], ref_seqs: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Refine clusters using multiple sequence alignments.
    
    Parameters:
    -----------
    clusters : Dict[str, List[str]]
        Dictionary of initial clusters
    goi_seqs : Dict[str, str]
        Dictionary of genes of interest sequences
    ref_seqs : Dict[str, str]
        Dictionary of reference sequences
        
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary of refined clusters
        
    Notes:
    ------
    - Runs multiple sequence alignments for each cluster
    - Uses M-Coffee meta-alignment to refine clusters
    - Logs refinement progress and results
    """
    logger.info("Refining clusters using multiple sequence alignments...")
    refined_clusters = defaultdict(list)
    
    for ref_id, cluster_members in clusters.items():
        if len(cluster_members) > 0:
            cluster_seqs = {ref_id: ref_seqs[ref_id]}
            for member in cluster_members:
                cluster_seqs[member] = goi_seqs[member]
            
            mafft_aln = run_alignment(cluster_seqs, 'mafft')
            tcoffee_aln = run_alignment(cluster_seqs, 'tcoffee')
            clustalw_aln = run_alignment(cluster_seqs, 'clustalw')
            
            meta_aln = run_mcoffee([mafft_aln, tcoffee_aln, clustalw_aln])
            
            aligned_seqs = load_sequences(meta_aln)
            for seq_id in cluster_members:
                if seq_id in aligned_seqs:
                    refined_clusters[ref_id].append(seq_id)
    
    logger.info("Cluster refinement complete")
    return refined_clusters

def discover_motifs(sequences: Dict[str, str], min_width: int = 6, max_width: int = 50) -> str:
    """
    Discover motifs using MEME.
    
    Parameters:
    -----------
    sequences : Dict[str, str]
        Dictionary of sequences for motif discovery
    min_width : int
        Minimum motif width (default: 6)
    max_width : int
        Maximum motif width (default: 50)
        
    Returns:
    --------
    str
        Path to the output motif file
        
    Notes:
    ------
    - Writes sequences to a temporary file
    - Runs MEME for motif discovery
    - Logs motif discovery progress and results
    """
    logger.info("Discovering motifs using MEME...")
    with NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as temp_in, \
         NamedTemporaryFile(suffix='.txt', delete=False) as temp_out:
        
        for seq_id, seq in sequences.items():
            temp_in.write(f">{seq_id}\n{seq}\n")
        temp_in.flush()
        
        sequence_type = "-protein" if any(c not in "ACGT" for c in next(iter(sequences.values()))) else "-dna"
        cmd = f"meme {temp_in.name} {sequence_type} -oc {temp_out.name} " + \
              f"-minw {min_width} -maxw {max_width} -nmotifs 5"
        subprocess.run(cmd, shell=True, check=True)
        
        logger.info(f"Motif discovery complete: {temp_out.name}")
        return temp_out.name

def score_motifs(sequence: str, motif_instances: List[motifs.Motif]) -> float:
    """
    Score a sequence against motif instances using PWM.
    
    Parameters:
    -----------
    sequence : str
        Sequence to score
    motif_instances : List[motifs.Motif]
        List of motif instances
        
    Returns:
    --------
    float
        Maximum PWM score for the sequence
        
    Notes:
    ------
    - Uses PWM scoring for motif instances
    - Calculates scores across the sequence
    - Logs scoring progress and results
    """
    logger.info("Scoring sequence against motif instances...")
    scores = []
    for motif in motif_instances:
        pwm = motif.counts.normalize(pseudocounts=0.5)
        scores.extend([pwm.calculate(sequence[i:i+len(motif)])
                      for i in range(len(sequence) - len(motif) + 1)
                      if i + len(motif) <= len(sequence)])
    
    max_score = max(scores) if scores else 0
    logger.info(f"Maximum motif score: {max_score}")
    return max_score

def refine_clusters_with_motifs(clusters: Dict[str, List[str]], goi_seqs: Dict[str, str], ref_seqs: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Refine clusters using both alignments and motif analysis.
    
    Parameters:
    -----------
    clusters : Dict[str, List[str]]
        Dictionary of initial clusters
    goi_seqs : Dict[str, str]
        Dictionary of genes of interest sequences
    ref_seqs : Dict[str, str]
        Dictionary of reference sequences
        
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary of refined clusters
        
    Notes:
    ------
    - Runs multiple sequence alignments for each cluster
    - Uses M-Coffee meta-alignment to refine clusters
    - Discovers motifs and scores sequences against motifs
    - Logs refinement progress and results
    """
    logger.info("Refining clusters using alignments and motif analysis...")
    refined_clusters = defaultdict(list)
    
    for ref_id, cluster_members in clusters.items():
        if len(cluster_members) > 0:
            cluster_seqs = {ref_id: ref_seqs[ref_id]}
            for member in cluster_members:
                cluster_seqs[member] = goi_seqs[member]
            
            mafft_aln = run_alignment(cluster_seqs, 'mafft')
            tcoffee_aln = run_alignment(cluster_seqs, 'tcoffee')
            clustalw_aln = run_alignment(cluster_seqs, 'clustalw')
            
            meta_aln = run_mcoffee([mafft_aln, tcoffee_aln, clustalw_aln])
            
            motif_file = discover_motifs(cluster_seqs)
            motif_instances = motifs.parse(motif_file, 'MEME')
            
            for seq_id in cluster_members:
                seq = goi_seqs[seq_id]
                motif_score = score_motifs(seq, motif_instances)
                
                if seq_id in load_sequences(meta_aln) and motif_score > 0.7:
                    refined_clusters[ref_id].append((seq_id, motif_score))
            
            refined_clusters[ref_id].sort(key=lambda x: x[1], reverse=True)
            refined_clusters[ref_id] = [x[0] for x in refined_clusters[ref_id]]
    
    logger.info("Cluster refinement with motifs complete")
    return refined_clusters

@log_performance_decorator
def process_motif_batch(batch_args):
    """Process a batch of sequences for motif discovery."""
    sequences, min_width, max_width = batch_args
    return discover_motifs(sequences, min_width, max_width)

def generate_markers(goi_seqs: Dict[str, str], ref_seqs: Dict[str, str], 
                    min_length: int = 8, max_length: int = 50, 
                    output_dir: str = "plots",
                    batch_size: int = 50) -> Dict[str, str]:
    """Optimized marker generation with parallel processing."""
    logger.info("Generating markers with parallel processing...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use joblib for caching expensive computations
    @joblib.Memory(output_dir).cache
    def cached_cluster_sequences(goi_seqs, ref_seqs):
        return cluster_sequences(goi_seqs, ref_seqs)
    
    initial_clusters = cached_cluster_sequences(goi_seqs, ref_seqs)
    plot_cluster_sizes(initial_clusters, f"{output_dir}/initial_cluster_sizes.png")
    
    # Process clusters in parallel
    markers = {}
    cluster_batches = []
    
    for ref_id, cluster_members in initial_clusters.items():
        if cluster_members:
            cluster_seqs = {member: goi_seqs[member] for member in cluster_members}
            if len(cluster_seqs) >= batch_size:
                # Split large clusters into batches
                seq_items = list(cluster_seqs.items())
                for i in range(0, len(seq_items), batch_size):
                    batch_dict = dict(seq_items[i:i+batch_size])
                    cluster_batches.append((batch_dict, min_length, max_length))
            else:
                cluster_batches.append((cluster_seqs, min_length, max_length))
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        motif_results = list(executor.map(process_motif_batch, cluster_batches))
    
    # ... rest of the generate_markers function ...
    
    return markers

def find_longest_common_substring(str1: str, str2: str) -> str:
    """
    Find the longest common substring between two sequences.
    
    Parameters:
    -----------
    str1 : str
        First sequence
    str2 : str
        Second sequence
        
    Returns:
    --------
    str
        Longest common substring
        
    Notes:
    ------
    - Uses dynamic programming to find the longest common substring
    - Logs progress and results
    """
    logger.info("Finding longest common substring...")
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end_pos = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
    
    longest_substring = str1[end_pos - max_length:end_pos]
    logger.info(f"Longest common substring: {longest_substring}")
    return longest_substring

def main():
    """
    Main execution function for marker generation pipeline.
    
    This function orchestrates the entire marker generation process:
    1. Parses command line arguments or snakemake parameters
    2. Validates input files and directories
    3. Loads and processes sequences
    4. Generates markers using various algorithms
    5. Outputs results and visualizations
    
    The function includes extensive error handling and logging to track
    progress and diagnose potential issues during execution.
    """
    if 'snakemake' in globals():
        args = argparse.Namespace(
            goi=snakemake.input.goi,
            reference=snakemake.input.reference,
            output=snakemake.output.markers,
            map_file=snakemake.output.map_file,
            plot_dir=snakemake.output.plot_dir
        )
    else:
        parser = argparse.ArgumentParser(description="Generate markers using reference database")
        parser.add_argument('-i', '--goi', required=True, help='Genes of interest FASTA file')
        parser.add_argument('-r', '--reference', required=True, help='Reference database FASTA file')
        parser.add_argument('-o', '--output', required=True, help='Output markers file')
        parser.add_argument('-m', '--map_file', required=True, help='Output mapping file')
        parser.add_argument('--plot-dir', default='plots', help='Directory for visualization outputs')
        args = parser.parse_args()

    if not os.path.exists(args.goi):
        raise FileNotFoundError(f"Input GOI file not found: {args.goi}")
    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference file not found: {args.reference}")

    os.makedirs(args.plot_dir, exist_ok=True)

    try:
        goi_seqs = load_sequences(args.goi)
        if not goi_seqs:
            raise ValueError(f"No sequences found in GOI file: {args.goi}")
            
        ref_seqs = load_sequences(args.reference)
        if not ref_seqs:
            raise ValueError(f"No sequences found in reference file: {args.reference}")

        markers = generate_markers(goi_seqs, ref_seqs, output_dir=args.plot_dir)
        
        if not markers:
            raise ValueError("No markers were generated")

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            for marker_id, seq in markers.items():
                f.write(f">{marker_id}\n{seq}\n")

        os.makedirs(os.path.dirname(args.map_file), exist_ok=True)
        with open(args.map_file, 'w') as f:
            for marker_id, seq in markers.items():
                goi_id = marker_id.split('_')[1]
                f.write(f"{marker_id}\t{goi_id}\n")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        start_time = time.time()
        logger.info("Starting marker generation pipeline")
        log_memory_usage()
        
        main()
        
        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(f"Pipeline completed successfully in {total_duration:.2f} seconds")
        log_memory_usage()
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)
        sys.exit(1)
