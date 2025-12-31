#!/usr/bin/env python3
# Portfolio-safe copy: biologically-informed k-mer utilities
# Identifiers generalized; no private data included.

import numpy as np
import pandas as pd
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, roc_curve, 
                           precision_recall_curve, auc, average_precision_score, 
                           confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import log
import mmh3  # MurmurHash library
import bitarray
import warnings
from collections import defaultdict
import subprocess
import tempfile
import datetime
import logging
import logging.handlers
from subprocess import CalledProcessError
from scipy.stats import entropy
import pywt
from scipy.fft import fft, fftfreq
from scipy import signal

# Add new imports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import hdbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from itertools import cycle
from tqdm import tqdm

# Optional imports with fallbacks
try:
    from Bio import (SeqIO, AlignIO, pairwise2, Entrez, Phylo, SearchIO, motifs)
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Data import CodonTable
    from Bio.Blast import NCBIWWW
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    from Bio.Phylo.PAML import yn00
    from Bio.Align import MultipleSeqAlignment
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

try:
    from cmdstanpy import CmdStanModel
    HAS_STAN = True
except ImportError:
    HAS_STAN = False

try:
    from ete3 import PhyloTree, TreeStyle, Tree
    HAS_ETE3 = True
except ImportError:
    HAS_ETE3 = False

try:
    import dendropy
    HAS_DENDROPY = True
except ImportError:
    HAS_DENDROPY = False

try:
    from scipy.stats import chi2_contingency
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from joblib import dump, load
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import dendropy
    from Bio.Phylo.TreeConstruction import _Matrix
    import numpy as np
    from scipy.stats import chi2
    HAS_PHYLO = True
except ImportError:
    HAS_PHYLO = False

# Add new imports for manifold learning
from sklearn.manifold import MDS, Isomap
from scipy.spatial.distance import squareform
import networkx as nx

# Update CONFIG structure for better organization
CONFIG = {
    'PATHS': {
            'log_file': os.path.join(os.getcwd(), 'logs', 'biological_kmers.log'),
            'model_dir': os.path.join(os.getcwd(), 'models'),
            'data_dir': os.path.join(os.getcwd(), 'data'),
            'output_dir': os.path.join(os.getcwd(), 'output'),
            'temp_dir': '/tmp/biological_kmers',
            'cache_dir': os.path.join(os.getcwd(), 'cache'),
            'databases': {
                'hmm': os.path.join(os.getcwd(), 'data', 'hmm', 'Pfam-A.hmm'),
                'biological_sequences': os.path.join(os.getcwd(), 'data', 'sequences', 'uniprot_sprot.fasta'),
                'contaminants': os.path.join(os.getcwd(), 'data', 'contaminants')
            }
        },
    'MODEL': {
        'kmer': {
            'size': 31,
            'min_length': 10,
            'max_length': 100,
            'gc_bounds': [0.3, 0.7],
            'complexity_threshold': 0.6
        },
        'training': {
            'batch_size': 1000,
            'random_state': 42,
            'validation_split': 0.2,
            'early_stopping_patience': 5,
            'max_epochs': 100,
            'sgd_learning_rate': 'optimal',
            'sgd_eta0': 0.01,
            'class_weight': 'balanced',
            'n_jobs': -1
        },
        'manifold': {
            'n_components': 10,
            'n_neighbors': 15,
            'metric': 'euclidean',
            'min_dist': 0.1,
            'conserved_threshold': 0.2,
            'neutral_threshold': 0.5
        }
    },
    'PROCESSING': {
        'num_threads': 8,
        'max_memory': '8G',
        'chunk_size': 10000,
        'cache_size': 1000,
        'timeout': 300,
        'retry_count': 3
    },
    'FILTERS': {
        'sequence': {
            'min_length': 100,
            'max_length': 10000,
            'min_gc': 0.3,
            'max_gc': 0.7,
            'exclude_n': True,
            'min_complexity': 0.6,
            'max_repeats': 0.5
        },
        'bloom': {
            'false_positive_rate': 0.01,
            'initial_capacity': 100000,
            'scaling_factor': 2,
            'num_hashes': 5
        },
        'quality': {
            'min_quality_score': 20,
            'max_n_content': 0.1,
            'trim_5_prime': 0,
            'trim_3_prime': 0
        }
    },
    'ANALYSIS': {
        'evolutionary': {
            'min_orthogroup_size': 3,
            'max_orthogroup_size': 1000,
            'min_sequence_identity': 0.3,
            'synteny_window': 5000,
            'duplication_threshold': 0.7,
            'selection_threshold': 0.05,
            'conservation_window': 10
        },
        'spectral': {
            'min_period': 2,
            'max_period': 500,
            'wavelet_level': 5,
            'wavelet_type': 'db1',
            'frequency_threshold': 0.05
        },
        'clustering': {
            'min_cluster_size': 5,
            'min_samples': 5,
            'metric': 'euclidean',
            'algorithm': 'auto',
            'leaf_size': 30
        },
        'tools': {
            'blast': {
                'evalue': 1e-5,
                'max_target_seqs': 500,
                'word_size': 11,
                'gapped': True,
                'dust': True
            },
            'hmmer': {
                'evalue': 1e-3,
                'cpu': 4,
                'domE': 1e-5,
                'incE': 0.01
            },
            'augustus': {
                'species': 'human',
                'strand': 'both',
                'genemodel': 'complete',
                'proteinprofile': ''
            },
            'repeatmasker': {
                'engine': 'crossmatch',
                'nolow': True,
                'norna': True,
                'gccalc': True
            }
        }
    },
    'LOGGING': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_size': 10485760,  # 10MB
        'backup_count': 5,
        'console_level': 'INFO',
        'log_to_file': True,
        'log_to_console': True,
        'capture_warnings': True
    },
    'VISUALIZATION': {
        'style': 'seaborn',
        'palette': 'viridis',
        'dpi': 300,
        'fig_size': (12, 8),
        'save_format': 'png',
        'interactive': False
    }
}

class ConfigManager:
    """Configuration management class with validation and type checking."""
    
    def __init__(self, config: dict):
        self._config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values and types."""
        required_sections = ['PATHS', 'MODEL', 'PROCESSING', 'FILTERS', 'ANALYSIS', 'LOGGING']
        for section in required_sections:
            if (section not in self._config) and (section != 'VISUALIZATION'):
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate paths
        for path_key, path in self._config['PATHS'].items():
            if isinstance(path, str) and not path_key.endswith('_dir'):
                if not os.path.exists(path):
                    logging.warning(f"Path does not exist: {path}")
    
    def get(self, *keys, default=None):
        """Get configuration value using nested keys."""
        value = self._config
        for key in keys:
            try:
                value = value[key]
            except (KeyError, TypeError):
                return default
        return value
    
    def set(self, value, *keys):
        """Set configuration value using nested keys."""
        config = self._config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
        self._validate_config()
    
    @property
    def paths(self):
        """Get paths configuration section."""
        return self._config['PATHS']
    
    @property
    def model(self):
        """Get model configuration section."""
        return self._config['MODEL']
    
    @property
    def processing(self):
        """Get processing configuration section."""
        return self._config['PROCESSING']

# Initialize configuration manager
config_manager = ConfigManager(CONFIG)

def setup_logging(log_file=None):
    """Configure logging with configuration values."""
    if log_file is None:
        log_file = config_manager.get('PATHS', 'log_file')
        
    logger = logging.getLogger('biological_kmers')
    logger.setLevel(getattr(logging, config_manager.get('LOGGING', 'level')))
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config_manager.get('LOGGING', 'file_size'),
        backupCount=config_manager.get('LOGGING', 'backup_count')
    )
    file_handler.setFormatter(logging.Formatter(config_manager.get('LOGGING', 'format')))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config_manager.get('LOGGING', 'console_level')))
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

class SubprocessError(Exception):
    """Custom exception for subprocess errors with detailed information"""
    def __init__(self, message, cmd=None, output=None, error=None):
        self.message = message
        self.cmd = cmd
        self.output = output
        self.error = error
        super().__init__(self.message)

def run_subprocess(cmd, timeout=300, check=True):
    """Run subprocess with robust error handling and logging.
    
    Args:
        cmd (str): Command to execute
        timeout (int): Maximum execution time in seconds
        check (bool): Whether to check return code
        
    Returns:
        subprocess.CompletedProcess: Completed process object
        
    Raises:
        SubprocessError: If process fails or times out
    """
    try:
        logger.debug(f"Running command: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=check
        )
        logger.debug(f"Command completed successfully: {cmd}")
        return result
        
    except TimeoutError:
        error_msg = f"Command timed out after {timeout}s: {cmd}"
        logger.error(error_msg)
        raise SubprocessError(error_msg, cmd=cmd)
        
    except CalledProcessError as e:
        error_msg = f"Command failed with exit code {e.returncode}: {cmd}"
        logger.error(error_msg)
        logger.error(f"Error output: {e.stderr}")
        raise SubprocessError(
            error_msg,
            cmd=cmd,
            output=e.output,
            error=e.stderr
        )
        
    except Exception as e:
        error_msg = f"Unexpected error running command: {cmd}"
        logger.error(error_msg, exc_info=True)
        raise SubprocessError(error_msg, cmd=cmd, error=str(e))

class BloomFilter:
    """Bloom filter for efficient sequence membership testing"""
    def __init__(self, size, num_hash_functions):
        self.size = size
        self.num_hash_functions = num_hash_functions
        self.bit_array = bitarray.bitarray(size)
        self.bit_array.setall(0)
    
    def add(self, item):
        for seed in range(self.num_hash_functions):
            index = mmh3.hash(str(item), seed) % self.size
            self.bit_array[index] = 1
    
    def __contains__(self, item):
        for seed in range(self.num_hash_functions):
            index = mmh3.hash(str(item), seed) % self.size
            if not self.bit_array[index]:
                return False
        return True

def create_bloom_filter(items):
    """Create a Bloom filter using configuration parameters."""
    bloom_config = config_manager.get('FILTERS', 'bloom')
    n = len(items)
    m = int(-n * log(bloom_config['false_positive_rate']) / (log(2) ** 2))
    k = int(m * log(2) / n)
    bloom = BloomFilter(m, k)
    for item in items:
        bloom.add(item)
    return bloom

def extract_features(sequence):
    """Extract features from k-mer sequence.
    
    Args:
        sequence (str): Input DNA sequence to analyze.
        
    Returns:
        list: List of numerical features including:
            - GC content (float): Ratio of G and C nucleotides
            - Sequence complexity (float): Ratio of unique to total nucleotides
    """
    if not sequence:
        raise ValueError("Sequence cannot be empty")
    
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    complexity = len(set(sequence)) / len(sequence)
    return [gc_content, complexity]

def get_nucleotide_freq(seq, n):
    """Calculate n-nucleotide frequencies in a sequence.
    
    Args:
        seq (str): Input DNA sequence.
        n (int): Length of nucleotide patterns to count (e.g., 2 for dinucleotides).
        
    Returns:
        dict: Mapping of n-nucleotide patterns to their frequencies.
            Keys are nucleotide patterns (str)
            Values are normalized frequencies (float)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if len(seq) < n:
        return {}
    
    freqs = {}
    for i in range(len(seq) - n + 1):
        nuc = seq[i:i+n]
        freqs[nuc] = freqs.get(nuc, 0) + 1
    return {k: v/(len(seq)-n+1) for k,v in freqs.items()}

def load_contaminant_db(contaminants_folder):
    """Load and index contaminant sequences from a folder.
    
    Args:
        contaminants_folder (str): Path to folder containing FASTA files.
        
    Returns:
        Union[BloomFilter, set]: Either a Bloom filter (for large datasets) or 
            set containing contaminant sequences. The Bloom filter is used for
            memory-efficient storage when database size exceeds 10000 sequences.
    
    Note:
        Supports both .fasta and .fa file extensions.
        Uses Bloom filter for large datasets to reduce memory usage.
    """
    contaminants = []
    for filename in os.listdir(contaminants_folder):
        if filename.endswith(('.fasta', '.fa')):
            filepath = os.path.join(contaminants_folder, filename)
            for record in SeqIO.parse(filepath, "fasta"):
                contaminants.append(str(record.seq))
    
    # Create Bloom filter index for large databases
    if len(contaminants) > 10000:
        return create_bloom_filter(contaminants)
    return set(contaminants)

def generate_random_sequences(n, length=100):
    """Generate random DNA sequences"""
    bases = ['A', 'T', 'G', 'C']
    return [''.join(np.random.choice(bases, length)) for _ in range(n)]

def is_contaminant(sequence, contaminants_folder="contaminants"):
    """Check if sequence matches known contaminants"""
    contaminant_db = load_contaminant_db(contaminants_folder)
    
    def create_feature_vectors_parallel(sequences):
        """Create feature vectors for multiple sequences in parallel"""
        with ProcessPoolExecutor() as executor:
            return list(executor.map(create_feature_vector, sequences))

    def create_feature_vector(sequence):
        """Create complete feature vector for a sequence"""
        features = extract_features(sequence)
        
        # Add di/tri/tetra-nucleotide frequencies
        for n in range(2,5):
            freq_dict = get_nucleotide_freq(sequence, n)
            features.extend(freq_dict.values())
        
        # Add entropy and codon bias
        features.append(get_entropy(sequence))
        features.append(get_codon_bias(sequence))
        
        return features
    return stan_model(sequence, contaminant_db)

def get_entropy(seq):
    """Calculate Shannon entropy of sequence"""
    counts = Counter(seq)
    probs = [count/len(seq) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs)

def get_codon_bias(seq):
    """Calculate simple codon usage bias"""
    if len(seq) < 3:
        return 0
    codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
    return len(set(codons)) / len(codons)

def get_repeat_content(seq):
    """Calculate fraction of sequence in repeats"""
    repeats = 0
    for i in range(len(seq)-3):
        for j in range(2, 6):  # Look for 2-5mer repeats
            if i + 2*j <= len(seq):
                if seq[i:i+j] == seq[i+j:i+2*j]:
                    repeats += j
                    break
    return repeats / len(seq) if len(seq) > 0 else 0

def get_kmer_entropy(seq, k=3):
    """Calculate k-mer entropy to measure sequence complexity"""
    if len(seq) < k:
        return 0
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    counts = Counter(kmers)
    probs = [count/len(kmers) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs)

def get_palindrome_content(seq):
    """Calculate fraction of sequence in palindromes"""
    palindromes = 0
    for i in range(len(seq)):
        for j in range(4, min(11, len(seq)-i)):  # Look for 4-10bp palindromes
            subseq = seq[i:i+j]
            if subseq == subseq[::-1]:
                palindromes += j
                break
    return palindromes / len(seq) if len(seq) > 0 else 0

def get_motif_scores(seq):
    """Calculate scores for common biological motifs"""
    motifs = {
        'tata_box': ['TATAAA', 'TATATA', 'TATATAT'],
        'splice_donor': ['GT', 'GC'],
        'splice_acceptor': ['AG'],
        'kozak': ['ACCATGG', 'GCCATGG'],
        'poly_a': ['AATAAA']
    }
    scores = {}
    for motif_type, patterns in motifs.items():
        score = sum(seq.count(pattern) for pattern in patterns)
        scores[f'{motif_type}_score'] = score / (len(seq) - 1) if len(seq) > 1 else 0
    return scores

def get_frame_bias(seq):
    """Calculate reading frame bias"""
    if len(seq) < 3:
        return 0
    frames = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
    frame_counts = Counter(frames)
    total = len(frames)
    return max(count/total for count in frame_counts.values()) if total > 0 else 0

def create_feature_vector(sequence, hmm_db_path=None):
    """Create comprehensive feature vector for sequence analysis.
    
    Args:
        sequence (str): Input DNA sequence.
        
    Returns:
        list: Numerical features including:
            - Basic features (GC content, complexity)
            - Nucleotide frequencies (di/tri/tetra-nucleotides)
            - Biological features (entropy, codon bias, repeats)
            - Motif scores (TATA box, splice sites, etc.)
            - External predictions (genes, tRNAs, etc.)
            - Evolutionary features (selection, phylogeny)
            - Relationship features (orthologs, paralogs)
    
    Note:
        This is the main feature extraction function that combines
        all feature types for comprehensive sequence analysis.
    """
    features = extract_features(sequence)
    
    # Add basic nucleotide frequencies
    for n in range(2,5):
        freq_dict = get_nucleotide_freq(sequence, n)
        features.extend(freq_dict.values())
    
    # Add basic features
    features.append(get_entropy(sequence))
    features.append(get_codon_bias(sequence))
    
    # Add new biological features
    features.append(get_repeat_content(sequence))
    features.append(get_palindrome_content(sequence))
    features.append(get_frame_bias(sequence))
    features.append(get_kmer_entropy(sequence, k=3))
    features.append(get_kmer_entropy(sequence, k=4))
    
    # Add motif scores
    motif_scores = get_motif_scores(sequence)
    features.extend(motif_scores.values())
    
    # Run external predictions
    predictions = run_external_predictions(sequence)
    
    # Add RepeatMasker features
    if (predictions['repeat_masker']):
        features.extend([
            predictions['repeat_masker']['repeat_count'],
            len(predictions['repeat_masker']['repeat_types']),
            np.mean(predictions['repeat_masker']['repeat_lengths'])
        ])
    else:
        features.extend([0, 0, 0])
    
    # Add gene prediction features
    if (predictions['gene_prediction']):
        features.extend([
            predictions['gene_prediction']['gene_count'],
            predictions['gene_prediction']['exon_count'],
            np.mean(predictions['gene_prediction']['cds_lengths'])
        ])
    else:
        features.extend([0, 0, 0])
    
    # Add tRNA features
    if (predictions['trna_scan']):
        features.extend([
            predictions['trna_scan']['trna_count'],
            len(predictions['trna_scan']['anticodon_types']),
            np.mean(predictions['trna_scan']['scores'])
        ])
    else:
        features.extend([0, 0, 0])
    
    # Add phylogenetic features
    features.extend([
        predictions['phyml_score'],
        predictions['selection_score'],
        predictions['convergence_prob']
    ])
    
    # Add evolutionary relationship features
    relationships = calculate_relationships(sequence)
    features.extend([
        relationships['orthogroup_size'],
        relationships['paralog_count'],
        relationships['duplication_score'],
        relationships['synteny_score'],
        relationships['cluster_density']
    ])
    
    # Add HMM features if database is provided
    if hmm_db_path and os.path.exists(hmm_db_path):
        hmm_features = get_hmm_features(sequence, hmm_db_path)
        features.extend([
            hmm_features['best_hmm_score'],
            hmm_features['hmm_hit_count'],
            hmm_features['mean_hmm_score'],
            -np.log10(hmm_features['min_hmm_evalue'] + 1e-300)  # Convert e-value to score
        ])
    
    # Add evolutionary model features
    features.extend(get_evolution_model_features(sequence))
    
    # Add enhanced phylogenetic features
    features.extend(get_enhanced_phylogenetic_features(sequence))
    
    # Add phylogenetic signal scores
    features.extend(get_phylogenetic_signal_scores(sequence, reference_sequences))
    
    # Add nucleotide entropy features
    entropy_metrics = calculate_nucleotide_entropy(sequence)
    features.extend([
        entropy_metrics['position_entropy'],
        entropy_metrics['position_entropy_std'],
        entropy_metrics['dinucleotide_entropy'],
        entropy_metrics['trinucleotide_entropy'],
        entropy_metrics['overall_entropy']
    ])
    
    # Add diversity metrics
    for k in [2, 3, 4]:  # Calculate for different k-mer sizes
        diversity_metrics = calculate_diversity_metrics(sequence, k=k)
        features.extend([
            diversity_metrics['simpson_diversity'],
            diversity_metrics['shannon_diversity'],
            diversity_metrics['evenness'],
            diversity_metrics['richness'],
            diversity_metrics['berger_parker']
        ])
    
    # Add constraint analysis features
    constraints = analyze_overlapping_constraints(sequence)
    features.extend([
        constraints['overlap_density'],
        constraints['frame_conservation'],
        constraints['regulatory_density'],
        constraints['structure_constraints']
    ])
    
    # Add motif scores
    motifs = find_regulatory_motifs(sequence)
    motif_score = sum(m['score'] for m in motifs) / len(sequence) if motifs else 0
    features.append(motif_score)
    
    # Add spectral analysis features
    features.extend(get_spectral_features(sequence))
    
    return features

def train_kmer_classifier(hmm_db_path=None):
    """Train classifier using configuration parameters."""
    if hmm_db_path is None:
        hmm_db_path = config_manager.get('PATHS', 'databases', 'hmm')
    
    model_config = config_manager.get('MODEL', 'training')
    clf = SGDClassifier(
        loss='log_loss',
        learning_rate=model_config['sgd_learning_rate'],
        eta0=model_config['sgd_eta0'],
        random_state=model_config['random_state']
    )
    model_handler = ModelPersistence()
    
    # Train incrementally
    # Add biological_sequences fasta path to CONFIG if not already present
    fasta_path = config_manager.get('PATHS', 'databases', 'biological_sequences')
    
    for i, (X_chunk, y_chunk) in enumerate(create_kmer_dataset(fasta_path, hmm_db_path=hmm_db_path)):
        if i == 0:
            # First chunk - fit the model with classes
            clf.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
        else:
            # Subsequent chunks - update the model
            clf.partial_fit(X_chunk, y_chunk)
        
        print(f"Processed chunk {i+1}")
    
    # Save trained model with metadata
    metadata = {
        'training_date': datetime.datetime.now().isoformat(),
        'data_source': fasta_path,
        'model_version': '1.0',
        'model_type': 'SGDClassifier', 
        'hmm_database': hmm_db_path if hmm_db_path else 'None'
    }
    model_path = model_handler.save_model(clf, metadata)
    
    return clf, model_path

def normalize_features(features, min_val=0, max_val=1):
    """Normalize features to prevent numerical instability"""
    if len(features) == 0:
        return features
    features = np.array(features)
    feature_range = np.max(features) - np.min(features)
    if feature_range == 0:
        return np.zeros_like(features)
    return min_val + (features - np.min(features)) * (max_val - np.min(features)) / feature_range

try:
    def stan_model(sequence, contaminant_db, prior_alpha=2, prior_beta=2):
        """Modified stan_model to handle both Bloom filter and regular databases"""
        if not HAS_STAN:
            return fallback_contamination_model(sequence, contaminant_db)
        
        # Handle empty database
        if isinstance(contaminant_db, set) and len(contaminant_db) < 2:
            return False
        
        # Prepare features with scaling, handling both set and BloomFilter
        if isinstance(contaminant_db, BloomFilter):
            features = np.array([1 if sequence in contaminant_db else 0])
        else:
            # Sample if regular database is too large
            db_sample = contaminant_db
            if isinstance(contaminant_db, set) and len(contaminant_db) > 10000:
                db_sample = np.random.choice(list(contaminant_db), 10000, replace=False)
            features = np.array([1 if c in sequence else 0 for c in db_sample])
        
        features = normalize_features(features)
        
        # Adjust sampling parameters based on database size
        n_samples = min(1000, max(100, len(contaminant_db) // 10))
        
        try:
            # Build and compile Stan model with more robust priors
            stan_code = """
            data {
                int<lower=0> N;
                vector[N] features;
                real<lower=0> alpha;
                real<lower=0> beta;
            }
            parameters {
                real<lower=0,upper=1> theta;
            }
            model {
                // More robust priors for extreme cases
                theta ~ beta(alpha + 1e-6, beta + 1e-6);
                features ~ bernoulli(theta + 1e-6);
            }
            """
            model = CmdStanModel(stan_code=stan_code)
            
            # Prepare data with bounds checking
            data = {
                'N': len(features), 
                'features': np.clip(features, 0, 1),  # Ensure valid range
                'alpha': max(prior_alpha, 1e-6),  # Prevent zero values
                'beta': max(prior_beta, 1e-6)
            }
            
            # Fit model with adaptive parameters
            fit = model.sample(
                data=data,
                iter_sampling=n_samples,
                iter_warmup=min(1000, n_samples),
                max_treedepth=8,
                adapt_delta=0.9
            )
            
            # Get result with bounds checking
            theta = fit.stan_variable('theta')
            if len(theta) == 0:
                return fallback_contamination_model(sequence, contaminant_db)
            
            return np.mean(theta) > 0.5
            
        except Exception as e:
            print(f"Stan model failed: {e}, falling back to logistic regression")
            return fallback_contamination_model(sequence, contaminant_db)
            
except ImportError:
    def stan_model(sequence, contaminant_db, prior_alpha=2, prior_beta=2):
        """Fallback version when Stan is not available"""
        return fallback_contamination_model(sequence, contaminant_db)

def fallback_contamination_model(sequence, contaminant_db):
    """Modified fallback model to handle Bloom filter"""
    features = extract_features(sequence)
    if not hasattr(fallback_contamination_model, 'model'):
        # Handle both set and BloomFilter cases
        if isinstance(contaminant_db, BloomFilter):
            X = np.array([[1 if sequence in contaminant_db else 0]])
        else:
            X = np.array([extract_features(s) for s in contaminant_db])
        y = np.ones(X.shape[0])
        
        # Add negative examples
        X_neg = np.array([extract_features(s) for s in generate_random_sequences(X.shape[0])])
        X = np.vstack([X, X_neg])
        y = np.hstack([y, np.zeros(X_neg.shape[0])])
        
        fallback_contamination_model.model = LogisticRegression()
        fallback_contamination_model.model.fit(X, y)
    
    return fallback_contamination_model.model.predict_proba([features])[0][1] > 0.5

def load_phylome(phylome_file, alignment_file):
    """Load phylogenetic tree and corresponding sequence alignment"""
    try:
        tree = PhyloTree(phylome_file, format=1)
        alignment = AlignIO.read(alignment_file, "fasta")
        return tree, alignment
    except Exception as e:
        warnings.warn(f"Error loading phylome: {e}")
        return None, None

def get_conservation_score(node, alignment):
    """Calculate conservation score for sequences in a given node"""
    if not node.is_leaf():
        seqs = [alignment[alignment.index(leaf.name)].seq for leaf in node.get_leaves()]
        if not seqs:
            return 0
        # Calculate column-wise conservation
        conservation = []
        for i in range(len(seqs[0])):
            column = [seq[i] for seq in seqs]
            conservation.append(len(set(column)) / len(column))
        return 1 - (sum(conservation) / len(conservation))
    return 0

def detect_evolutionary_events(sequence, reference, max_indel_size=10):
    """Detect potential evolutionary events in sequence compared to reference"""
    events = {
        'indels': [],
        'inversions': [],
        'duplications': [],
        'pseudogene_features': []
    }
    
    # Detect indels using pairwise alignment
    alignment = pairwise2.align.globalms(sequence, reference, 
                                       2, -1, -2, -0.5, 
                                       one_alignment_only=True)[0]
    
    # Process alignment to find indels
    seq_pos = ref_pos = 0
    for i, (seq_char, ref_char) in enumerate(zip(alignment[0], alignment[1])):
        if seq_char == '-' or ref_char == '-':
            if len(events['indels']) > 0 and i == events['indels'][-1][1] + 1:
                events['indels'][-1] = (events['indels'][-1][0], i)
            else:
                events['indels'].append((i, i))
    
    # Detect inversions
    for i in range(len(sequence)-10):
        for j in range(i+10, len(sequence)):
            subseq = sequence[i:j]
            if subseq == subseq[::-1]:
                events['inversions'].append((i, j))
    
    # Detect duplications
    for i in range(len(sequence)-5):
        for j in range(i+5, len(sequence)):
            subseq = sequence[i:j]
            if sequence.count(subseq) > 1:
                events['duplications'].append((i, j))
    
    # Check for pseudogene features
    events['pseudogene_features'] = detect_pseudogene_features(sequence)
    
    return events

def detect_pseudogene_features(sequence):
    """Detect features indicative of pseudogenization"""
    features = []
    
    # Look for premature stop codons
    for i in range(0, len(sequence)-2, 3):
        codon = sequence[i:i+3]
        if codon in ['TAA', 'TAG', 'TGA']:
            features.append(('premature_stop', i))
    
    # Look for frameshift mutations
    for i in range(len(sequence)-3):
        if sequence[i:i+3] in ['ATG']:  # Start codon
            frame_shifts = check_frame_shifts(sequence[i:])
            features.extend([('frameshift', i+pos) for pos in frame_shifts])
    
    return features

def check_frame_shifts(seq):
    """Check for potential frameshift mutations"""
    shifts = []
    canonical_length = len(seq) - (len(seq) % 3)
    
    for i in range(0, canonical_length-3, 3):
        codon = seq[i:i+3]
        next_codon = seq[i+3:i+6]
        
        # Check for insertions/deletions that disrupt reading frame
        if codon in ['ATG', 'GTG'] and next_codon not in ['ATG', 'GTG']:
            if any(base not in 'ATGC' for base in next_codon):
                shifts.append(i+3)
    
    return shifts

def extract_confident_kmers(tree, alignment, k_size=31, confidence_threshold=0.8):
    """Extract k-mers accounting for evolutionary events"""
    confident_kmers = []
    event_tolerant_kmers = defaultdict(list)
    
    for node in tree.traverse():
        if not node.is_leaf():
            seqs = [alignment[alignment.index(leaf.name)].seq for leaf in node.get_leaves()]
            if not seqs:
                continue
                
            # Get consensus and check conservation
            conservation = get_conservation_score(node, alignment)
            if (conservation >= confidence_threshold):
                consensus = get_consensus_sequence(seqs)
                
                # Extract k-mers considering evolutionary events
                for i in range(len(consensus) - k_size + 1):
                    kmer = consensus[i:i+k_size]
                    if 'N' not in kmer and '-' not in kmer:
                        # Check for evolutionary events in this region
                        events = detect_evolutionary_events(kmer, consensus[i:i+k_size])
                        
                        if events['indels'] or events['inversions'] or events['duplications']:
                            event_tolerant_kmers[kmer].append((conservation, events))
                        else:
                            confident_kmers.append((kmer, conservation))
    
    # Process event-tolerant k-mers
    for kmer, data in event_tolerant_kmers.items():
        avg_conservation = sum(c for c, _ in data) / len(data)
        if avg_conservation >= confidence_threshold:
            confident_kmers.append((kmer, avg_conservation))
    
    return confident_kmers

def get_consensus_sequence(sequences):
    """Get consensus sequence handling gaps and variations"""
    if not sequences:
        return ""
        
    consensus = []
    for i in range(len(sequences[0])):
        column = [seq[i] for seq in sequences if i < len(seq)]
        counts = Counter(column)
        # Handle gaps more permissively
        if '-' in counts and counts['-'] <= len(column) // 2:
            counts.pop('-')
        consensus.append(counts.most_common(1)[0][0])
    
    return ''.join(consensus)

def load_known_biological_elements():
    """Load well-known biological sequence elements"""
    elements = {
        'ori': [],  # Origins of replication
        'trna': [],  # tRNA sequences
        'crispr': [],  # CRISPR repeats
        'promoter': [],  # Core promoter elements
        'terminator': [],  # Transcription terminators
        'rrna': []  # rRNA sequences
    }
    
    # Common bacterial ori sequences
    elements['ori'].extend([
        "ATTTAATGATCCACAG",  # E. coli oriC
        "TTATCCACAGGGCAGG",  # B. subtilis oriC
        "TTGTCCACACTGGAAG"   # M. tuberculosis oriC
    ])
    
    # Common tRNA sequence patterns
    elements['trna'].extend([
        "GGTTCGATCC",  # tRNA 3' end
        "GTGGCNNAGT",  # tRNA D-arm
        "TTCGANNTC"    # tRNA anticodon arm
    ])
    
    # CRISPR repeat sequences
    elements['crispr'].extend([
        "GTTTTAGAGCTATGCT",  # SpCas9
        "ATTCCATTTAAAAAGG",  # Common bacterial repeat
        "GTTCCATTTGAAAGGG"   # Common archaeal repeat
    ])
    
    # Core promoter elements
    elements['promoter'].extend([
        "TATAAT",  # Bacterial -10 box
        "TTGACA",  # Bacterial -35 box
        "TATAAA"   # Eukaryotic TATA box
    ])
    
    # Transcription terminators
    elements['terminator'].extend([
        "GCGCCCGGCT",  # rho-independent terminator
        "TGCCTGGCAC",  # intrinsic terminator
    ])
    
    # rRNA conserved regions
    elements['rrna'].extend([
        "GAATTGACGGAAG",  # 16S rRNA
        "CACACCGCCCGT",   # 23S rRNA
        "GCTGGCACCAGA"    # 5S rRNA
    ])
    
    return elements

def extract_known_kmers(sequence, k_size=31):
    """Extract k-mers around known biological elements"""
    known_elements = load_known_biological_elements()
    known_kmers = []
    
    for element_type, patterns in known_elements.items():
        for pattern in patterns:
            # Find all occurrences of the pattern
            start = 0
            while True:
                pos = sequence.find(pattern, start)
                if pos == -1:
                    break
                    
                # Extract k-mer centered on the pattern
                start_pos = max(0, pos - (k_size - len(pattern))//2)
                end_pos = min(len(sequence), start_pos + k_size)
                
                if end_pos - start_pos == k_size:
                    kmer = sequence[start_pos:end_pos]
                    known_kmers.append((kmer, element_type))
                
                start = pos + 1
    
    return known_kmers

# Update create_kmer_dataset to use event-aware extraction
def create_kmer_dataset(fasta_file="uniprot_sprot.fasta", phylome_file=None, alignment_file=None, 
                       k_size=config_manager.get('MODEL', 'kmer', 'size'), 
                       negative_sample_size=None, event_tolerance=True, 
                       include_known_elements=True, chunk_size=config_manager.get('MODEL', 'training', 'batch_size'), 
                       hmm_db_path=config_manager.get('PATHS', 'databases', 'hmm')):
    """Create dataset with evolutionary event awareness"""
    X = []
    y = []
    biological_kmers = set()
    
    if (phylome_file and alignment_file and event_tolerance):
        tree, alignment = load_phylome(phylome_file, alignment_file)
        if tree and alignment:
            confident_kmers = extract_confident_kmers(tree, alignment, k_size)
            for kmer, confidence in confident_kmers:
                features = create_feature_vector(kmer, hmm_db_path=hmm_db_path)
                # Add evolutionary event features
                events = detect_evolutionary_events(kmer, kmer)  # Self-comparison for feature extraction
                features.extend([
                    len(events['indels']),
                    len(events['inversions']),
                    len(events['duplications']),
                    len(events['pseudogene_features'])
                ])
                X.append(features)
                y.append(1)
                biological_kmers.add(kmer)
    
    # Process additional sequences from fasta file
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        for i in range(len(seq) - k_size + 1):
            kmer = seq[i:i+k_size]
            if 'N' not in kmer:  # Skip k-mers with ambiguous bases
                biological_kmers.add(kmer)
                features = extract_features(kmer)
                X.append(features)
                y.append(1)  # 1 for biological
    
    # Process known biological elements
    if include_known_elements:
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq)
            known_kmers = extract_known_kmers(seq, k_size)
            
            for kmer, element_type in known_kmers:
                if 'N' not in kmer and kmer not in biological_kmers:
                    biological_kmers.add(kmer)
                    features = create_feature_vector(kmer, hmm_db_path=hmm_db_path)
                    # Add element type as feature
                    element_features = [1 if et == element_type else 0 
                                     for et in load_known_biological_elements().keys()]
                    features.extend(element_features)
                    X.append(features)
                    y.append(1)
    
    def generate_random_kmers(n, existing_kmers):
        bases = ['A', 'T', 'G', 'C']
        while n > 0:
            kmer = ''.join(np.random.choice(bases, k_size))
            if kmer not in existing_kmers:
                yield kmer
                n -= 1

    def parallel_feature_extraction(sequences):
        with ProcessPoolExecutor() as executor:
            return list(executor.map(extract_features, sequences))

    # Use generator for random k-mers
    random_kmers = list(generate_random_kmers(negative_sample_size, biological_kmers))
    random_features = parallel_feature_extraction(random_kmers)
    
    X.extend(random_features)
    y.extend([0] * len(random_features))
    
    return np.array(X), np.array(y)

def predict_kmer(clf, sequence, k_size=31):
    """Predict if a k-mer is biological or artifact"""
    # Validate k-mer length
    if len(sequence) != k_size:
        raise ValueError(f"K-mer must be of length {k_size}")
    
    # Validate nucleotide characters
    valid_bases = set('ATGC')
    if not set(sequence.upper()).issubset(valid_bases):
        raise ValueError("K-mer contains invalid characters. Only A,T,G,C are allowed")
        
    features = create_feature_vector(sequence)
    prediction = clf.predict([features])[0]
    return "Biological" if prediction == 1 else "Artifact"

# Update get_feature_names to include new features
def format_feature_name(name):
    """Format feature names for better visualization"""
    # Handle nucleotide frequency names
    if 'mer_' in name:
        n, bases = name.split('mer_')
        return f"{n}-Nucleotide Freq: {bases}"
    
    # Handle special cases
    name_map = {
        'gc_content': 'GC Content',
        'complexity': 'Sequence Complexity',
        'entropy': 'Shannon Entropy',
        'codon_bias': 'Codon Usage Bias',
        'repeat_content': 'Repeat Content',
        'palindrome_content': 'Palindrome Content',
        'frame_bias': 'Reading Frame Bias',
        'kmer_entropy_3': '3-mer Entropy',
        'kmer_entropy_4': '4-mer Entropy',
        'tata_box_score': 'TATA Box Signal',
        'splice_donor_score': 'Splice Donor Site',
        'splice_acceptor_score': 'Splice Acceptor Site',
        'kozak_score': 'Kozak Sequence',
        'poly_a_score': 'Poly-A Signal',
        'indel_count': 'Indel Events',
        'inversion_count': 'Inversion Events',
        'duplication_count': 'Duplication Events',
        'pseudogene_feature_count': 'Pseudogene Features',
        'repeat_count': 'Repeat Count',
        'repeat_type_diversity': 'Repeat Type Diversity',
        'mean_repeat_length': 'Mean Repeat Length',
        'gene_count': 'Gene Count',
        'exon_count': 'Exon Count',
        'mean_cds_length': 'Mean CDS Length',
        'trna_count': 'tRNA Count',
        'anticodon_diversity': 'Anticodon Diversity',
        'mean_trna_score': 'Mean tRNA Score',
        'phylogenetic_signal': 'Phylogenetic Signal',
        'selection_pressure': 'Selection Pressure',
        'convergence_probability': 'Convergence Probability',
        'orthogroup_size': 'Orthogroup Size',
        'paralog_count': 'Paralog Count',
        'duplication_score': 'Duplication Score',
        'synteny_score': 'Synteny Score',
        'sequence_cluster_density': 'Sequence Cluster Density',
        'best_hmm_score': 'Best HMM Score',
        'hmm_hit_count': 'HMM Hit Count',
        'mean_hmm_score': 'Mean HMM Score',
        'min_hmm_evalue': 'Min HMM E-value',
        'position_entropy': 'Position Entropy',
        'position_entropy_std': 'Position Entropy Std',
        'dinucleotide_entropy': 'Dinucleotide Entropy',
        'trinucleotide_entropy': 'Trinucleotide Entropy',
        'overall_entropy': 'Overall Entropy',
        '2mer_simpson_diversity': '2-mer Simpson Diversity',
        '2mer_shannon_diversity': '2-mer Shannon Diversity',
        '2mer_evenness': '2-mer Evenness',
        '2mer_richness': '2-mer Richness',
        '2mer_berger_parker': '2-mer Berger Parker',
        '3mer_simpson_diversity': '3-mer Simpson Diversity',
        '3mer_shannon_diversity': '3-mer Shannon Diversity',
        '3mer_evenness': '3-mer Evenness',
        '3mer_richness': '3-mer Richness',
        '3mer_berger_parker': '3-mer Berger Parker',
        '4mer_simpson_diversity': '4-mer Simpson Diversity',
        '4mer_shannon_diversity': '4-mer Shannon Diversity',
        '4mer_evenness': '4-mer Evenness',
        '4mer_richness': '4-mer Richness',
        '4mer_berger_parker': '4-mer Berger Parker',
        'overlap_density': 'Overlap Density',
        'frame_conservation': 'Frame Conservation',
        'regulatory_density': 'Regulatory Density',
        'structure_constraints': 'Structure Constraints',
        'motif_score': 'Motif Score',
        'mean_period': 'Mean Period',
        'spectral_entropy': 'Spectral Entropy',
        'codon_periodicity': 'Codon Periodicity',
        'helix_periodicity': 'Helix Periodicity',
        'nucleosome_periodicity': 'Nucleosome Periodicity',
        'wavelet_energy_mean': 'Wavelet Energy Mean',
        'wavelet_energy_std': 'Wavelet Energy Std',
        'wavelet_entropy_mean': 'Wavelet Entropy Mean',
        'wavelet_entropy_max': 'Wavelet Entropy Max',
        'wavelet_entropy_min': 'Wavelet Entropy Min'
    }
    return name_map.get(name, name.replace('_', ' ').title())

def get_feature_names():
    """Get descriptive feature names including evolutionary events"""
    names = ['gc_content', 'complexity']
    
    # Add nucleotide frequency names
    for n in range(2,5):
        bases = ['A', 'T', 'G', 'C']
        names.extend([f'{n}mer_{"".join(combo)}' for combo in product(bases, repeat=n)])
    
    # Add basic feature names
    names.extend(['entropy', 'codon_bias'])
    
    # Add biological feature names
    names.extend([
        'repeat_content',
        'palindrome_content',
        'frame_bias',
        'kmer_entropy_3',
        'kmer_entropy_4',
        'tata_box_score',
        'splice_donor_score',
        'splice_acceptor_score',
        'kozak_score',
        'poly_a_score'
    ])
    
    # Add evolutionary event features
    names.extend([
        'indel_count',
        'inversion_count',
        'duplication_count',
        'pseudogene_feature_count'
    ])
    
    # Add biological element type features
    names.extend([f'{element}_sequence' 
                 for element in load_known_biological_elements().keys()])
    
    # Add prediction feature names
    names.extend([
        'repeat_count',
        'repeat_type_diversity',
        'mean_repeat_length',
        'gene_count',
        'exon_count',
        'mean_cds_length',
        'trna_count',
        'anticodon_diversity',
        'mean_trna_score',
        'phylogenetic_signal',
        'selection_pressure',
        'convergence_probability'
    ])
    
    # Add evolutionary relationship feature names
    names.extend([
        'orthogroup_size',
        'paralog_count',
        'duplication_score',
        'synteny_score',
        'sequence_cluster_density'
    ])
    
    # Add HMM feature names
    names.extend([
        'best_hmm_score',
        'hmm_hit_count',
        'mean_hmm_score',
        'min_hmm_evalue'
    ])
    
    # Add entropy feature names
    names.extend([
        'position_entropy',
        'position_entropy_std',
        'dinucleotide_entropy',
        'trinucleotide_entropy',
        'overall_entropy'
    ])
    
    # Add diversity feature names for each k-mer size
    for k in [2, 3, 4]:
        names.extend([
            f'{k}mer_simpson_diversity',
            f'{k}mer_shannon_diversity',
            f'{k}mer_evenness',
            f'{k}mer_richness',
            f'{k}mer_berger_parker'
        ])
    
    # Add constraint analysis features
    names.extend([
        'overlap_density',
        'frame_conservation',
        'regulatory_density',
        'structure_constraints',
        'motif_score'
    ])
    
    # Add spectral feature names
    names.extend([
        'mean_period',
        'spectral_entropy',
        'codon_periodicity',
        'helix_periodicity',
        'nucleosome_periodicity',
        'wavelet_energy_mean',
        'wavelet_energy_std',
        'wavelet_entropy_mean',
        'wavelet_entropy_max',
        'wavelet_entropy_min'
    ])
    
    # Format all names
    return [format_feature_name(name) for name in names]

def visualize_results(clf, X_test, y_test, feature_names=None):
    """Visualize model results and feature importance."""
    if feature_names is None:
        feature_names = get_feature_names()
    
    # Plot feature importance with improved formatting
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance, x='Importance', y='Feature')
    plt.title('Feature Importance in Biological Sequence Classification', pad=20)
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Add value labels
    for i, v in enumerate(importance['Importance']):
        plt.text(v, i, f'{v:.3f}', va='center')
    
    plt.show()

def visualize_phylogenetic_kmers(tree, confident_kmers):
    """Visualize k-mer distribution across phylogenetic tree"""
    ts = TreeStyle()
    ts.show_leaf_name = True
    
    def layout(node):
        if not node.is_leaf():
            node_kmers = [k for k, _ in confident_kmers if k in node.get_leaf_names()]
            if node_kmers:
                node.img_style["size"] = 10
                node.img_style["fgcolor"] = "red"
    
    ts.layout_fn = layout
    tree.show(tree_style=ts)

def evaluate_model(clf, X_test, y_test, feature_names=None):
    """Perform comprehensive model evaluation with visualizations.
    
    Args:
        clf: Trained classifier object
        X_test: Test feature matrix
        y_test: True test labels
        feature_names (list, optional): List of feature names for plotting
        
    Returns:
        dict: Evaluation metrics including:
            - accuracy (float): Overall accuracy
            - roc_auc (float): Area under ROC curve
            - pr_auc (float): Area under Precision-Recall curve
            - class_report (dict): Per-class precision, recall, F1
            - confusion_matrix (array): Confusion matrix
    
    Note:
        Creates visualization plots for:
        - ROC curve
        - Precision-Recall curve
        - Confusion matrix
        - Feature importance
    """
    # Get predictions and probabilities
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot ROC curve
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('Receiver Operating Characteristic')
    axes[0, 0].legend(loc="lower right")
    
    # Plot Precision-Recall curve
    axes[0, 1].plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.2f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc="lower left")
    
    # Plot confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-bio', 'Bio'],
                yticklabels=['Non-bio', 'Bio'],
                ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    
    # Plot feature importance
    if feature_names is None:
        feature_names = get_feature_names()
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)  # Show top 10
    
    sns.barplot(data=importance, x='Importance', y='Feature', ax=axes[1, 1])
    axes[1, 1].set_title('Top 10 Feature Importance')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    print("\nDetailed Classification Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    print("\nClass-wise Metrics:")
    for label in ['0', '1']:
        print(f"\nClass {label}:")
        print(f"Precision: {class_report[label]['precision']:.3f}")
        print(f"Recall: {class_report[label]['recall']:.3f}")
        print(f"F1-score: {class_report[label]['f1-score']:.3f}")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'class_report': class_report,
        'confusion_matrix': conf_matrix
    }

def run_external_predictions(sequence):
    """Run external bioinformatics tools for sequence analysis.
    
    Args:
        sequence (str): Input DNA sequence.
        
    Returns:
        dict: Results from multiple tools:
            - repeat_masker: Repeat element analysis
            - gene_prediction: Gene structure predictions
            - trna_scan: tRNA predictions
            - phyml_score: Phylogenetic analysis score
            - selection_score: Natural selection analysis
            - convergence_prob: Bayesian convergence probability
    
    Note:
        Requires external tools:
        - RepeatMasker
        - AUGUSTUS
        - tRNAscan-SE
        - PhyML
        - PAML
    """
    results = {
        'repeat_masker': None,
        'gene_prediction': None,
        'trna_scan': None,
        'phyml_score': None,
        'selection_score': None,
        'convergence_prob': None
    }
    
    # Create temporary file for sequence
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:
        temp_file.write(f">seq\n{sequence}\n")
        temp_file.flush()
        
        # Run RepeatMasker
        try:
            result = run_subprocess(f"RepeatMasker -noint {temp_file.name}")
            results['repeat_masker'] = parse_repeatmasker_output(result.stdout)
        except SubprocessError as e:
            logger.error("RepeatMasker analysis failed", exc_info=True)
            results['repeat_masker'] = None
        
        # Run AUGUSTUS gene prediction
        try:
            result = run_subprocess(f"augustus --species=human {temp_file.name}")
            results['gene_prediction'] = parse_augustus_output(result.stdout)
        except SubprocessError as e:
            logger.error("AUGUSTUS analysis failed", exc_info=True)
            results['gene_prediction'] = None
        
        # Run tRNAscan-SE
        try:
            result = run_subprocess(f"tRNAscan-SE {temp_file.name}")
            results['trna_scan'] = parse_trnascan_output(result.stdout)
        except SubprocessError as e:
            logger.error("tRNAscan-SE analysis failed", exc_info=True)
            results['trna_scan'] = None
    
    # Run phylogenetic analysis
    try:
        results['phyml_score'] = analyze_phylogenetic_signal(sequence)
    except Exception as e:
        logger.error("PhyML analysis failed", exc_info=True)
        results['phyml_score'] = 0.0
    
    try:
        results['selection_score'] = detect_selection_footprints(sequence)
    except Exception as e:
        logger.error("Selection analysis failed", exc_info=True)
        results['selection_score'] = 1.0
    
    try:
        results['convergence_prob'] = assess_bayesian_convergence(sequence)
    except Exception as e:
        logger.error("Convergence analysis failed", exc_info=True)
        results['convergence_prob'] = 0.5
    
    return results

def parse_repeatmasker_output(output):
    """Parse RepeatMasker output and extract features"""
    features = {
        'repeat_count': 0,
        'repeat_types': set(),
        'repeat_lengths': []
    }
    
    for line in output.split('\n'):
        if line.startswith('SW'):
            features['repeat_count'] += 1
            parts = line.split()
            if len(parts) > 10:
                features['repeat_types'].add(parts[10])
                features['repeat_lengths'].append(int(parts[6]) - int(parts[5]))
    
    return features

def parse_augustus_output(output):
    """Parse AUGUSTUS gene prediction output"""
    features = {
        'gene_count': 0,
        'exon_count': 0,
        'cds_lengths': []
    }
    
    for line in output.split('\n'):
        if '\tgene\t' in line:
            features['gene_count'] += 1
        elif '\tCDS\t' in line:
            parts = line.split('\t')
            if len(parts) > 4:
                features['cds_lengths'].append(int(parts[4]) - int(parts[3]))
    
    return features

def parse_trnascan_output(output):
    """Parse tRNAscan-SE output"""
    features = {
        'trna_count': 0,
        'anticodon_types': set(),
        'scores': []
    }
    
    for line in output.split('\n'):
        if not line.startswith('#'):
            parts = line.split()
            if len(parts) > 4:
                features['trna_count'] += 1
                features['anticodon_types'].add(parts[4])
                features['scores'].append(float(parts[8]))
    
    return features

def analyze_phylogenetic_signal(sequence, reference_db="nr"):
    """Analyze phylogenetic signal using BLAST and PhyML"""
    try:
        # BLAST search
        result_handle = NCBIWWW.qblast("blastn", reference_db, sequence)
        blast_records = NCBIWWW.parse(result_handle)
        
        # Collect homologous sequences
        homologs = []
        for record in blast_records:
            for alignment in record.alignments:
                homologs.append(alignment.seq)
        
        if not homologs:
            return 0.0
        
        # Build distance matrix
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(homologs)
        
        # Construct tree
        constructor = DistanceTreeConstructor()
        tree = constructor.build_tree(dm)
        
        # Calculate phylogenetic signal
        return calculate_phylogenetic_signal(tree)
        
    except Exception as e:
        print(f"Phylogenetic analysis failed: {e}")
        return 0.0

def detect_selection_footprints(sequence):
    """Detect footprints of selection using dN/dS ratio"""
    try:
        # Create temporary alignment file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:
            temp_file.write(f">seq\n{sequence}\n")
            temp_file.flush()
            
            # Run PAML
            result = run_subprocess(f"codeml {temp_file.name}")
            return parse_paml_output(result.stdout)
            
    except SubprocessError as e:
        logger.error("PAML analysis failed", exc_info=True)
        return 1.0  # Neutral evolution as default

def assess_bayesian_convergence(sequence):
    """Assess Bayesian convergence in phylogenetic trees"""
    try:
        # Create DendroPy tree from sequence
        tree = dendropy.Tree.get_from_string(
            sequence,
            schema="newick",
            preserve_underscores=True
        )
        
        # Calculate convergence probability
        prob = calculate_convergence_probability(tree)
        return prob
        
    except Exception as e:
        print(f"Convergence analysis failed: {e}")
        return 0.5  # Neutral convergence as default

def calculate_phylogenetic_signal(tree):
    """Calculate phylogenetic signal using Blomberg's K"""
    try:
        # Convert tree to matrix form
        matrix = tree.cophenetic_matrix()
        
        # Calculate Blomberg's K
        k_stat = calculate_blomberg_k(matrix)
        return k_stat
        
    except Exception as e:
        print(f"Signal calculation failed: {e}")
        return 0.0

def calculate_convergence_probability(tree):
    """Calculate Bayesian convergence probability"""
    try:
        # Get tree statistics
        tree_len = tree.length()
        node_ages = [node.age for node in tree.nodes()]
        
        # Calculate convergence probability
        prob = 1.0 - chi2_contingency(node_ages)[1]
        return prob
        
    except Exception as e:
        print(f"Convergence calculation failed: {e}")
        return 0.5

def analyze_evolutionary_relationships(sequence, database_path):
    """Analyze evolutionary relationships of a sequence.
    
    Args:
        sequence (str): Input DNA sequence.
        database_path (str): Path to FASTA file containing reference sequences.
        
    Returns:
        dict: Dictionary containing:
            - orthogroup_size (int): Number of sequences in orthogroup
            - paralog_count (int): Number of identified paralogs
            - duplication_score (float): Score indicating duplication events
            - synteny_score (float): Measure of synteny conservation
            - cluster_density (float): Density of sequence clustering
    
    Note:
        Combines multiple evolutionary analyses including:
        - Orthogroup identification
        - Paralog detection
        - Synteny analysis
        - Sequence space clustering
    """
    features = {
        'orthogroup_size': 0,
        'paralog_count': 0,
        'duplication_score': 0.0,
        'synteny_score': 0.0,
        'cluster_density': 0.0
    }
    
    try:
        # Find orthogroups using OrthoFinder-like approach
        orthogroup = identify_orthogroup(sequence, database_path)
        features['orthogroup_size'] = len(orthogroup)
        
        # Analyze paralogs and duplications
        paralogs = find_paralogs(sequence, orthogroup)
        features['paralog_count'] = len(paralogs)
        
        # Calculate duplication scores
        features['duplication_score'] = calculate_duplication_score(sequence, paralogs)
        
        # Analyze synteny
        features['synteny_score'] = analyze_synteny(sequence, orthogroup)
        
        # Calculate sequence space clustering
        features['cluster_density'] = calculate_cluster_density(sequence, orthogroup)
        
        return features
    
    except Exception as e:
        print(f"Evolutionary analysis failed: {e}")
        return features

def identify_orthogroup(sequence, database_path):
    """Identify orthogroup members using sequence similarity"""
    orthogroup = []
    
    try:
        # Load sequence database
        sequences = list(SeqIO.parse(database_path, "fasta"))
        
        # Calculate similarity matrix
        distances = calculate_distance_matrix(sequence, sequences)
        
        # Cluster sequences
        Z = linkage(distances, method='average')
        clusters = fcluster(Z, t=0.7, criterion='distance')
        
        # Get sequences in same cluster as query
        query_cluster = clusters[0]
        orthogroup = [seq for i, seq in enumerate(sequences) 
                     if clusters[i] == query_cluster]
        
    except Exception as e:
        print(f"Orthogroup identification failed: {e}")
    
    return orthogroup

def find_paralogs(sequence, orthogroup):
    """Identify paralogs within orthogroup"""
    paralogs = []
    
    try:
        # Create sequence similarity network
        G = nx.Graph()
        
        # Add nodes and edges based on sequence similarity
        for seq1 in orthogroup:
            for seq2 in orthogroup:
                if seq1 != seq2:
                    similarity = calculate_sequence_similarity(seq1, seq2)
                    if similarity > 0.3:  # Similarity threshold
                        G.add_edge(seq1.id, seq2.id, weight=similarity)
        
        # Find connected components (paralog groups)
        paralog_groups = list(nx.connected_components(G))
        
        # Get paralogs for query sequence
        for group in paralog_groups:
            if sequence in group:
                paralogs = list(group - {sequence})
                break
                
    except Exception as e:
        print(f"Paralog identification failed: {e}")
    
    return paralogs

def calculate_duplication_score(sequence, paralogs):
    """Calculate duplication score based on synteny and similarity"""
    try:
        if not paralogs:
            return 0.0
            
        similarities = []
        for paralog in paralogs:
            # Calculate sequence similarity
            sim = calculate_sequence_similarity(sequence, paralog)
            
            # Calculate synteny conservation
            syn = calculate_synteny_conservation(sequence, paralog)
            
            # Combine metrics
            similarities.append((sim + syn) / 2)
        
        return np.mean(similarities)
        
    except Exception as e:
        print(f"Duplication score calculation failed: {e}")
        return 0.0

def analyze_synteny(sequence, orthogroup):
    """Analyze syntenic relationships"""
    try:
        # Create synteny blocks
        blocks = identify_synteny_blocks(sequence, orthogroup)
        
        # Calculate synteny conservation score
        conservation = calculate_synteny_conservation_score(blocks)
        
        return conservation
        
    except Exception as e:
        print(f"Synteny analysis failed: {e}")
        return 0.0

def calculate_cluster_density(sequence, orthogroup):
    """Calculate sequence space clustering density"""
    try:
        # Convert sequences to numerical vectors
        vectors = [seq_to_vector(seq) for seq in orthogroup]
        
        # Calculate pairwise distances
        distances = pdist(vectors)
        
        # Calculate clustering coefficient
        if len(distances) > 0:
            return 1.0 / (1.0 + np.mean(distances))
        return 0.0
        
    except Exception as e:
        print(f"Cluster density calculation failed: {e}")
        return 0.0

def seq_to_vector(sequence):
    """Convert sequence to numerical vector for clustering"""
    # Use k-mer frequencies as features
    k = 3
    kmers = [''.join(p) for p in product('ATGC', repeat=k)]
    counts = Counter([''.join(sequence[i:i+k]) 
                     for i in range(len(sequence)-k+1)])
    return [counts.get(kmer, 0) for kmer in kmers]

def calculate_distance_matrix(query_sequence, sequences):
    """Calculate pairwise distances between query and database sequences"""
    distances = []
    for seq in sequences:
        distance = calculate_sequence_distance(query_sequence, str(seq.seq))
        distances.append(distance)
    return np.array(distances)

def calculate_sequence_distance(seq1, seq2):
    """Calculate normalized edit distance between sequences"""
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    matches = sum(a == b for a, b in zip(alignment[0], alignment[1]))
    return 1 - (matches / max(len(seq1), len(seq2)))

def calculate_sequence_similarity(seq1, seq2):
    """Calculate sequence similarity score"""
    return 1 - calculate_sequence_distance(seq1, seq2)

def calculate_synteny_conservation(seq1, seq2):
    """Calculate synteny conservation score between sequences"""
    # Get neighboring genes
    neighbors1 = get_neighboring_genes(seq1)
    neighbors2 = get_neighboring_genes(seq2)
    
    # Calculate conservation
    shared = len(set(neighbors1) & set(neighbors2))
    total = len(set(neighbors1) | set(neighbors2))
    
    return shared / total if total > 0 else 0

def get_neighboring_genes(sequence, window=5000):
    """Get neighboring genes within window size"""
    neighbors = []
    try:
        # Run gene prediction
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:
            temp_file.write(f">seq\n{sequence}\n")
            temp_file.flush()
            
            result = run_subprocess(f"augustus --species=human {temp_file.name}")
            
            for line in result.stdout.split('\n'):
                if '\tgene\t' in line:
                    parts = line.split('\t')
                    if len(parts) > 4:
                        start = int(parts[3])
                        if start <= window:  # Within window
                            neighbors.append(parts[8])  # Gene ID
                            
    except SubprocessError as e:
        logger.error("Gene prediction failed", exc_info=True)
        
    return neighbors

def identify_synteny_blocks(sequence, orthogroup):
    """Identify syntenic blocks between sequences"""
    blocks = []
    try:
        for ortho_seq in orthogroup:
            # Get gene order
            genes1 = get_neighboring_genes(sequence)
            genes2 = get_neighboring_genes(str(ortho_seq.seq))
            
            # Find conserved blocks
            i = j = 0
            current_block = []
            while i < len(genes1) and j < len(genes2):
                if genes1[i] == genes2[j]:
                    current_block.append(genes1[i])
                    i += 1
                    j += 1
                else:
                    if len(current_block) >= 3:  # Minimum block size
                        blocks.append(current_block)
                    current_block = []
                    i += 1
                    j += 1
                    
            if len(current_block) >= 3:
                blocks.append(current_block)
                
    except Exception as e:
        print(f"Synteny block identification failed: {e}")
        
    return blocks

def calculate_synteny_conservation_score(blocks):
    """Calculate conservation score from synteny blocks"""
    if not blocks:
        return 0.0
    
    # Calculate average block size and coverage
    avg_size = np.mean([len(block) for block in blocks])
    total_genes = sum(len(block) for block in blocks)
    
    # Normalize score
    return (avg_size * total_genes) / 10000  # Arbitrary scaling

def calculate_blomberg_k(matrix):
    """Calculate Blomberg's K statistic"""
    try:
        # Convert distance matrix to phylogenetic signal
        n = len(matrix)
        if n < 3:
            return 0.0
            
        # Calculate phylogenetic variance
        phylo_var = np.sum(matrix) / (n * (n-1))
        
        # Calculate trait variance
        trait_var = np.var(np.diagonal(matrix))
        
        # Calculate K statistic
        if trait_var == 0:
            return 0.0
            
        k = phylo_var / trait_var
        
        return max(0.0, min(1.0, k))  # Bound between 0 and 1
        
    except Exception as e:
        print(f"Blomberg's K calculation failed: {e}")
        return 0.0

def parse_paml_output(output):
    """Parse PAML output to get dN/dS ratio"""
    try:
        for line in output.split('\n'):
            if 'dN/dS' in line:
                return float(line.split('=')[1].strip())
        return 1.0
    except:
        return 1.0

def sequence_generator(fasta_file, chunk_size=1000):
    """Generator to stream sequences from FASTA file.
    
    Args:
        fasta_file (str): Path to FASTA file
        chunk_size (int): Number of sequences to yield at once
        
    Yields:
        list: Chunk of sequences
    """
    chunk = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        chunk.append(str(record.seq))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def kmer_generator(sequence, k_size=31):
    """Generator for k-mers from a sequence.
    
    Args:
        sequence (str): Input sequence
        k_size (int): K-mer size
        
    Yields:
        str: K-mer sequence
    """
    for i in range(len(sequence) - k_size + 1):
        kmer = sequence[i:i+k_size]
        if 'N' not in kmer:
            yield kmer

def feature_generator(sequences, k_size=31):
    """Generator for feature vectors from sequences.
    
    Args:
        sequences (iterable): Input sequences
        k_size (int): K-mer size
        
    Yields:
        tuple: (features, label)
    """
    for sequence in sequences:
        for kmer in kmer_generator(sequence, k_size):
            features = create_feature_vector(kmer)
            yield features, 1  # 1 for biological sequence

def create_kmer_dataset(fasta_file, phylome_file=None, alignment_file=None, k_size=config_manager.get('MODEL', 'kmer', 'size'), 
                       negative_sample_size=None, event_tolerance=True, 
                       include_known_elements=True, chunk_size=config_manager.get('MODEL', 'training', 'batch_size'), hmm_db_path=config_manager.get('PATHS', 'databases', 'hmm')):
    """Create dataset with streaming data handling"""
    X, y = [], []
    biological_kmers = BloomFilter(1000000, 5)  # Use Bloom filter instead of set
    
    # Process sequences in chunks
    for chunk in sequence_generator(fasta_file, chunk_size):
        # Process each sequence in the chunk
        for features, label in feature_generator(chunk, k_size):
            X.append(features)
            y.append(label)
            
        # Generate negative examples for this chunk
        chunk_size = len(chunk)
        random_sequences = generate_random_sequences(chunk_size, k_size)
        for seq in random_sequences:
            if seq not in biological_kmers:  # Use Bloom filter for membership test
                features = create_feature_vector(seq, hmm_db_path=hmm_db_path)
                X.append(features)
                y.append(0)
                
        # Process in batches to avoid memory issues
        if len(X) >= chunk_size * 2:
            X_chunk = np.array(X)
            y_chunk = np.array(y)
            yield X_chunk, y_chunk
            X, y = [], []
    
    # Yield remaining data
    if X:
        yield np.array(X), np.array(y)

def train_kmer_classifier(hmm_db_path=None):
    """Train classifier using configuration parameters."""
    if hmm_db_path is None:
        hmm_db_path = config_manager.get('PATHS', 'databases', 'hmm')
    
    model_config = config_manager.get('MODEL', 'training')
    clf = SGDClassifier(
        loss='log_loss',
        learning_rate=model_config['sgd_learning_rate'],
        eta0=model_config['sgd_eta0'],
        random_state=model_config['random_state']
    )
    model_handler = ModelPersistence()
    
    # Train incrementally
    for i, (X_chunk, y_chunk) in enumerate(create_kmer_dataset("biological_sequences.fasta", hmm_db_path=hmm_db_path)):
        if i == 0:
            # First chunk - fit the model with classes
            clf.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
        else:
            # Subsequent chunks - update the model
            clf.partial_fit(X_chunk, y_chunk)
        
        print(f"Processed chunk {i+1}")
    
    # Save trained model with metadata
    metadata = {
        'training_date': datetime.datetime.now().isoformat(),
        'data_source': "biological_sequences.fasta",
        'model_version': '1.0',
        'model_type': 'SGDClassifier',
        'hmm_database': hmm_db_path if hmm_db_path else 'None'
    }
    model_path = model_handler.save_model(clf, metadata)
    
    return clf, model_path

def predict_kmer(sequence, k_size=31, model_path=None):
    """Predict using saved model"""
    model_handler = ModelPersistence()
    
    try:
        # Load model (latest if path not specified)
        clf = model_handler.load_model(model_path) if model_path else model_handler.load_latest_model()
        
        # Validate input
        if len(sequence) != k_size:
            raise ValueError(f"K-mer must be of length {k_size}")
        
        valid_bases = set('ATGC')
        if not set(sequence.upper()).issubset(valid_bases):
            raise ValueError("K-mer contains invalid characters. Only A,T,G,C are allowed")
        
        # Make prediction
        features = create_feature_vector(sequence)
        prediction = clf.predict([features])[0]
        return "Biological" if prediction == 1 else "Artifact"
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

class ModelPersistence:
    """Handle model saving and loading with versioning"""
    
    def __init__(self, model_dir=config_manager.get('PATHS', 'model_dir')):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model, metadata=None):
        """Save model with timestamp and metadata"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"kmer_model_{timestamp}.joblib")
        
        # Prepare metadata
        meta = {
            'timestamp': timestamp,
            'n_features': len(model.coef_[0]),  # Changed from feature_importances_
            'model_type': 'SGDClassifier',
            'user_metadata': metadata or {}
        }
        
        # Save model and metadata
        dump({'model': model, 'metadata': meta}, model_path)
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_latest_model(self):
        """Load most recent model"""
        models = sorted([f for f in os.listdir(self.model_dir) if f.startswith("kmer_model_")])
        if not models:
            raise FileNotFoundError("No saved models found")
        
        latest_model = os.path.join(self.model_dir, models[-1])
        saved_data = load(latest_model)
        print(f"Loaded model from {latest_model}")
        print(f"Model metadata: {saved_data['metadata']}")
        return saved_data['model']
    
    def load_model(self, model_path):
        """Load specific model by path"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        saved_data = load(model_path)
        print(f"Loaded model from {model_path}")
        print(f"Model metadata: {saved_data['metadata']}")
        return saved_data['model']

def parse_hmm_database(hmm_db_path):
    """Parse HMM database and extract profile information.
    
    Args:
        hmm_db_path (str): Path to HMM database file
        
    Returns:
        dict: Dictionary of HMM profiles with their metadata
    """
    profiles = {}
    try:
        # Use hmmfetch --index to create index if not present
        if not os.path.exists(f"{hmm_db_path}.h3i"):
            subprocess.run(['hmmfetch', '--index', hmm_db_path], check=True)
            
        # Parse HMM database
        with open(hmm_db_path) as hmm_file:
            for record in SearchIO.parse(hmm_file, 'hmmer3-text'):
                profiles[record.id] = {
                    'length': record.seq_len,
                    'description': record.description,
                    'cutoffs': {
                        'ga': record.ga,  # Gathering threshold
                        'tc': record.tc,  # Trusted cutoff
                        'nc': record.nc   # Noise cutoff
                    } if hasattr(record, 'ga') else None
                }
        return profiles
    except Exception as e:
        logger.error(f"Error parsing HMM database: {e}")
        return {}

def scan_sequence_with_hmm(sequence, hmm_db_path):
    """Scan sequence against HMM database using HMMER.
    
    Args:
        sequence (str): Input sequence
        hmm_db_path (str): Path to HMM database
        
    Returns:
        list: List of HMM hits with scores
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:
            temp_file.write(f">seq\n{sequence}\n")
            temp_file.flush()
            
            # Run hmmscan
            result = run_subprocess(
                f"hmmscan --cut_ga --domtblout /dev/stdout {hmm_db_path} {temp_file.name}",
                timeout=60
            )
            
            # Parse results
            hits = []
            for line in result.stdout.split('\n'):
                if line and not line.startswith('#'):
                    fields = line.split()
                    if len(fields) >= 13:
                        hits.append({
                            'hmm_name': fields[0],
                            'hmm_acc': fields[1],
                            'score': float(fields[7]),
                            'e_value': float(fields[6])
                        })
            return hits
    except Exception as e:
        logger.error(f"Error running HMM scan: {e}")
        return []

def get_hmm_features(sequence, hmm_db_path):
    """Extract HMM-based features from sequence.
    
    Args:
        sequence (str): Input sequence
        hmm_db_path (str): Path to HMM database
        
    Returns:
        dict: HMM features including best hit scores and coverage
    """
    hits = scan_sequence_with_hmm(sequence, hmm_db_path)
    
    features = {
        'best_hmm_score': 0.0,
        'hmm_hit_count': len(hits),
        'mean_hmm_score': 0.0,
        'min_hmm_evalue': float('inf')
    }
    
    if hits:
        scores = [hit['score'] for hit in hits]
        evalues = [hit['e_value'] for hit in hits]
        features.update({
            'best_hmm_score': max(scores),
            'mean_hmm_score': sum(scores) / len(scores),
            'min_hmm_evalue': min(evalues)
        })
    
    return features

def calculate_relationships(sequence, database_path=None):
    """Calculate evolutionary relationship features.
    
    Args:
        sequence (str): Input DNA sequence
        database_path (str): Optional path to sequence database
        
    Returns:
        dict: Dictionary containing relationship features:
            - orthogroup_size (int): Size of orthogroup
            - paralog_count (int): Number of paralogs
            - duplication_score (float): Duplication score
            - synteny_score (float): Synteny conservation score
            - cluster_density (float): Sequence clustering density
    """
    default_features = {
        'orthogroup_size': 0,
        'paralog_count': 0,
        'duplication_score': 0.0,
        'synteny_score': 0.0,
        'cluster_density': 0.0
    }
    
    if not database_path:
        return default_features
        
    try:
        # Get evolutionary relationships if database is provided
        return analyze_evolutionary_relationships(sequence, database_path)
    except Exception as e:
        logger.error(f"Failed to calculate relationships: {e}")
        return default_features

def get_sequence_space_features(kmer):
    """Calculate features based on sequence space clustering properties.
    
    Args:
        kmer (str): Input k-mer sequence
        
    Returns:
        list: Features capturing evolutionary constraints:
            - Local density: Density of similar sequences
            - Mutational robustness: Resistance to mutations
            - Compositional complexity: Sequence complexity measures
            - Evolutionary conservation: Conservation signals
    """
    features = []
    
    # Calculate local density in sequence space
    neighbors = get_hamming_neighbors(kmer, max_dist=2)
    local_density = len([n for n in neighbors if is_viable_sequence(n)])
    features.append(local_density / len(neighbors))
    
    # Calculate mutational robustness
    robustness = calculate_mutational_robustness(kmer)
    features.append(robustness)
    
    # Add compositional complexity
    features.extend([
        get_linguistic_complexity(kmer),
        get_compression_complexity(kmer),
        get_entropy_density(kmer)
    ])
    
    # Add evolutionary conservation signals
    features.extend([
        get_codon_conservation(kmer),
        get_motif_density(kmer),
        get_structural_constraints(kmer)
    ])
    
    return features

def get_hamming_neighbors(sequence, max_dist=2):
    """Get all sequences within Hamming distance."""
    neighbors = set()
    bases = ['A', 'T', 'G', 'C']
    
    def recursive_neighbors(seq, pos, dist):
        if dist > max_dist:
            return
        neighbors.add(seq)
        if pos >= len(seq):
            return
        
        orig_base = seq[pos]
        for base in bases:
            if base != orig_base:
                new_seq = seq[:pos] + base + seq[pos+1:]
                recursive_neighbors(new_seq, pos+1, dist+1)
        recursive_neighbors(seq, pos+1, dist)
    
    recursive_neighbors(sequence, 0, 0)
    return neighbors

def calculate_mutational_robustness(sequence):
    """Calculate sequence resistance to mutations."""
    neighbors = get_hamming_neighbors(sequence, max_dist=1)
    viable_count = sum(1 for n in neighbors if is_viable_sequence(n))
    return viable_count / len(neighbors)

def is_viable_sequence(sequence):
    """Check if sequence is likely viable based on biological constraints."""
    # Check basic composition
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    if not (0.2 <= gc_content <= 0.8):
        return False
    
    # Check for extreme repetition
    if max(sequence.count(base) for base in 'ATGC') > len(sequence) * 0.8:
        return False
    
    # Check for known biological patterns
    if has_biological_patterns(sequence):
        return True
    
    return None  # Uncertain

def has_biological_patterns(sequence):
    """Check for presence of known biological sequence patterns."""
    patterns = {
        'start_codons': ['ATG', 'GTG', 'TTG'],
        'stop_codons': ['TAA', 'TAG', 'TGA'],
        'splice_sites': ['GT', 'AG'],
        'promoter_elements': ['TATA', 'CAAT'],
        'regulatory_motifs': ['GCGC', 'CGCG', 'TGCA']
    }
    
    for pattern_type, motifs in patterns.items():
        if any(motif in sequence for motif in motifs):
            return True
    return False

def get_linguistic_complexity(sequence):
    """Calculate linguistic sequence complexity."""
    observed_kmers = set()
    total_possible = 0
    
    for k in range(1, min(len(sequence), 5) + 1):
        observed = set(sequence[i:i+k] for i in range(len(sequence)-k+1))
        observed_kmers.update(observed)
        total_possible += min(4**k, len(sequence)-k+1)
    
    return len(observed_kmers) / total_possible

def get_compression_complexity(sequence):
    """Estimate sequence complexity using compression."""
    import zlib
    compressed = zlib.compress(sequence.encode())
    return len(compressed) / len(sequence)

def get_entropy_density(sequence, window=5):
    """Calculate local entropy density."""
    entropies = []
    for i in range(len(sequence) - window + 1):
        window_seq = sequence[i:i+window]
        counts = Counter(window_seq)
        entropy = -sum((c/window * log(c/window)) for c in counts.values())
        entropies.append(entropy)
    return sum(entropies) / len(entropies)

def get_codon_conservation(sequence):
    """Calculate conservation of codon usage patterns."""
    if len(sequence) < 3:
        return 0
    
    codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
    codon_freqs = Counter(codons)
    
    # Compare to expected frequencies in biological sequences
    expected_freqs = {
        'ATG': 0.02, 'TGG': 0.01, 'TAA': 0.002, 'TAG': 0.001, 'TGA': 0.002
        # Add more expected frequencies...
    }
    
    deviation = 0
    for codon, freq in codon_freqs.items():
        expected = expected_freqs.get(codon, 0.015)  # Default frequency
        deviation += abs(freq/len(codons) - expected)
    
    return 1 / (1 + deviation)

def get_motif_density(sequence):
    """Calculate density of biological sequence motifs."""
    motifs = load_known_biological_elements()
    total_matches = 0
    
    for element_type, patterns in motifs.items():
        for pattern in patterns:
            total_matches += sequence.count(pattern)
    
    return total_matches / len(sequence)

def get_structural_constraints(sequence):
    """Evaluate potential structural constraints."""
    # GC content in different frames
    frame_gc = []
    for i in range(3):
        frame = sequence[i::3]
        if frame:
            gc = (frame.count('G') + frame.count('C')) / len(frame)
            frame_gc.append(gc)
    
    # Measure frame bias
    if frame_gc:
        return np.std(frame_gc)
    return 0

def create_feature_vector(sequence, include_clustering=True):
    """Create feature vector with emphasis on sequence space clustering."""
    features = []
    
    # Add sequence space clustering features
    if include_clustering:
        features.extend(get_sequence_space_features(sequence))
    
    # Add basic sequence features
    features.extend([
        get_gc_content(sequence),
        get_entropy(sequence),
        get_linguistic_complexity(sequence)
    ])
    
    # Add evolutionary signal features
    features.extend([
        get_codon_bias(sequence),
        get_motif_density(sequence),
        get_structural_constraints(sequence)
    ])
    
    # Add evolutionary model features
    
    clf.fit(X, y)
    return clf

def generate_random_kmers(n, k=31):
    """Generate random k-mers with similar base composition."""
    bases = ['A', 'T', 'G', 'C']
    return [''.join(np.random.choice(bases, k)) for _ in range(n)]

if __name__ == "__main__":
    # Example usage for biological sequence analysis
    logger = setup_logging()
    logger.info("Starting biological k-mer analysis")
    
    try:
        # Get database path from config
        fasta_path = config_manager.get('PATHS', 'databases', 'biological_sequences')
        
        # Train and evaluate model
        clf, metrics = train_kmer_classifier(fasta_path)
        
        # Save model if performance is good
        if metrics['f1'] > 0.8:
            model_handler = ModelPersistence()
            model_handler.save_model(clf, metadata={'metrics': metrics})
            logger.info("Model saved successfully")
            
    except Exception as e:
        logger.error("Analysis failed", exc_info=True)
        raise

def analyze_evolutionary_model(sequences, modeltest_path="modeltest-ng"):
    """Analyze sequence evolution model using ModelTest-NG.
    
    Args:
        sequences (list): List of DNA sequences
        modeltest_path (str): Path to ModelTest-NG executable
    
    Returns:
        dict: Model parameters including:
            - best_model (str): Best-fit evolutionary model
            - parameters (dict): Model parameters (rates, frequencies)
            - aic_score (float): Akaike Information Criterion score
            - bic_score (float): Bayesian Information Criterion score
            - lnl (float): Log-likelihood score
    """
    try:
        # Create temporary alignment file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:
            for i, seq in enumerate(sequences):
                temp_file.write(f">seq_{i}\n{seq}\n")
            temp_file.flush()
            
            # Run ModelTest-NG
            cmd = f"{modeltest_path} -i {temp_file.name} -d nt -p 4 --topology ml"
            result = run_subprocess(cmd)
            
            # Parse results
            return parse_modeltest_output(result.stdout)
            
    except SubprocessError as e:
        logger.error("ModelTest analysis failed", exc_info=True)
        return None

def parse_modeltest_output(output):
    """Parse ModelTest-NG output to extract model parameters."""
    results = {
        'best_model': None,
        'parameters': {},
        'aic_score': None,
        'bic_score': None,
        'lnl': None
    }
    
    try:
        lines = output.split('\n')
        for line in lines:
            if 'Best model according to ' in line:
                if 'AIC' in line:
                    results['best_model'] = line.split()[-1]
                    
            elif 'Model parameters:' in line:
                param_section = True
                continue
                
            elif param_section and ':' in line:
                key, value = line.split(':')
                results['parameters'][key.strip()] = float(value.strip())
                
            elif 'lnL:' in line:
                results['lnl'] = float(line.split(':')[1].strip())
                
            elif 'AIC score:' in line:
                results['aic_score'] = float(line.split(':')[1].strip())
                
            elif 'BIC score:' in line:
                results['bic_score'] = float(line.split(':')[1].strip())
                
        return results
        
    except Exception as e:
        logger.error(f"Error parsing ModelTest output: {e}")
        return None

def get_evolution_model_features(kmer, reference_sequences=None):
    """Extract features based on evolutionary model fit.
    
    Args:
        kmer (str): Input k-mer sequence
        reference_sequences (list): Optional list of reference sequences
        
    Returns:
        list: Features including:
            - Model complexity score
            - Parameter richness
            - Model fit scores
            - Rate heterogeneity
    """
    features = []
    
    if reference_sequences is None:
        # Generate sequence variants through mutations
        variants = list(get_hamming_neighbors(kmer, max_dist=3))
        variants = [v for v in variants if is_viable_sequence(v)]
    else:
        variants = reference_sequences
    
    if len(variants) < 4:  # Need minimum sequences for model fitting
        return [0.0] * 5  # Return default features
        
    # Get model test results
    model_results = analyze_evolutionary_model(variants)
    
    if model_results:
        # Model complexity (based on number of parameters)
        features.append(len(model_results['parameters']))
        
        # Parameter richness (variation in rates)
        if 'rates' in model_results['parameters']:
            rates = model_results['parameters']['rates']
            features.append(np.std(rates))
        else:
            features.append(0.0)
            
        # Model fit scores
        features.append(-model_results['aic_score'])  # Negative because lower is better
        features.append(-model_results['bic_score'])
        
        # Rate heterogeneity (alpha parameter if available)
        features.append(model_results['parameters'].get('alpha', 0.0))
        
    else:
        features.extend([0.0] * 5)
    
    return features

def create_feature_vector(sequence, include_clustering=True, reference_sequences=None):
    """Create comprehensive feature vector including evolutionary model analysis."""
    features = []
    
    # Add sequence space clustering features
    if include_clustering:
        features.extend(get_sequence_space_features(sequence))
    
    # Add basic sequence features
    features.extend([
        get_gc_content(sequence),
        get_entropy(sequence),
        get_linguistic_complexity(sequence)
    ])
    
    # Add evolutionary signal features
    features.extend([
        get_codon_bias(sequence),
        get_motif_density(sequence),
        get_structural_constraints(sequence)
    ])
    
    # Add evolutionary model features
    features.extend(get_evolution_model_features(sequence, reference_sequences))
    
    # Add enhanced phylogenetic features
    features.extend(get_enhanced_phylogenetic_features(sequence, reference_sequences))
    
    # Add phylogenetic signal scores
    features.extend(get_phylogenetic_signal_scores(sequence, reference_sequences))
    
    # Add nucleotide entropy features
    entropy_metrics = calculate_nucleotide_entropy(sequence)
    features.extend([
        entropy_metrics['position_entropy'],
        entropy_metrics['position_entropy_std'],
        entropy_metrics['dinucleotide_entropy'],
        entropy_metrics['trinucleotide_entropy'],
        entropy_metrics['overall_entropy']
    ])
    
    # Add diversity metrics
    for k in [2, 3, 4]:  # Calculate for different k-mer sizes
        diversity_metrics = calculate_diversity_metrics(sequence, k=k)
        features.extend([
            diversity_metrics['simpson_diversity'],
            diversity_metrics['shannon_diversity'],
            diversity_metrics['evenness'],
            diversity_metrics['richness'],
            diversity_metrics['berger_parker']
        ])
    
    return features

def calculate_parsimony_score(sequence, reference_sequences):
    """Calculate parsimony score based on sequence changes."""
    if not reference_sequences:
        return 0.0
    
    changes = []
    for ref in reference_sequences:
        change_count = sum(1 for a, b in zip(sequence, ref) if a != b)
        changes.append(change_count)
    
    return min(changes)  # Most parsimonious score

def calculate_maximum_likelihood(sequence, reference_sequences):
    """Calculate maximum likelihood score for sequence evolution."""
    try:
        # Create alignment
        alignment = MultipleSeqAlignment([
            SeqRecord(Seq(seq), id=f"seq_{i}") 
            for i, seq in enumerate([sequence] + reference_sequences)
        ])
        
        # Calculate likelihood using PhyML
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(alignment)
        constructor = DistanceTreeConstructor()
        tree = constructor.build_tree(alignment)
        
        # Get likelihood score
        if hasattr(tree, 'log_likelihood'):
            return tree.log_likelihood
        else:
            return -sum(dm) / len(dm)  # Approximate score
            
    except Exception as e:
        logger.error(f"Maximum likelihood calculation failed: {e}")
        return 0.0

def calculate_bayesian_posterior(sequence, reference_sequences):
    """Calculate Bayesian posterior probability for sequence."""
    try:
        # Run MrBayes-like analysis using simplified model
        priors = {
            'gc_prior': 0.5,
            'transition_prior': 0.33,
            'transversion_prior': 0.17
        }
        
        # Calculate likelihood
        likelihood = calculate_maximum_likelihood(sequence, reference_sequences)
        
        # Calculate prior probability
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        prior = np.exp(-abs(gc_content - priors['gc_prior']))
        
        # Approximate posterior
        posterior = likelihood * prior
        return posterior
        
    except Exception as e:
        logger.error(f"Bayesian analysis failed: {e}")
        return 0.0

def calculate_phylogenetic_signal(sequence, window_size=5):
    """Calculate phylogenetic signal using sliding windows."""
    signals = []
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        
        # Calculate conservation score
        conservation = len(set(window)) / window_size
        
        # Calculate position weight
        pos_weight = 1.0 - (abs(i - len(sequence)/2) / (len(sequence)/2))
        
        signals.append(conservation * pos_weight)
    
    return np.mean(signals)

def analyze_kmer_cooccurrence(sequence, k=3, window=10):
    """Analyze k-mer co-occurrence patterns."""
    cooccurrence = defaultdict(float)
    
    # Generate all possible k-mers
    kmers = [''.join(p) for p in product('ATGC', repeat=k)]
    
    # Calculate co-occurrence frequencies
    for i in range(len(sequence) - window + 1):
        window_seq = sequence[i:i+window]
        window_kmers = set(window_seq[j:j+k] for j in range(len(window_seq)-k+1))
        
        for kmer1, kmer2 in combinations(window_kmers, 2):
            cooccurrence[(kmer1, kmer2)] += 1
    
    # Normalize frequencies
    total = sum(cooccurrence.values()) or 1
    return {k: v/total for k, v in cooccurrence.items()}

def calculate_positional_bias(sequence, k=3):
    """Calculate positional bias of k-mers."""
    positions = defaultdict(list)
    
    for i in range(len(sequence)-k+1):
        kmer = sequence[i:i+k]
        positions[kmer].append(i / (len(sequence)-k))
    
    # Calculate position variance for each k-mer
    bias_scores = {kmer: np.var(pos) for kmer, pos in positions.items()}
    return bias_scores

def get_enhanced_phylogenetic_features(sequence, reference_sequences=None):
    """Get comprehensive phylogenetic features."""
    features = []
    
    # Generate reference sequences if not provided
    if reference_sequences is None:
        reference_sequences = list(get_hamming_neighbors(sequence, max_dist=2))
        reference_sequences = [s for s in reference_sequences if is_viable_sequence(s)]
    
    # Calculate parsimony score
    features.append(calculate_parsimony_score(sequence, reference_sequences))
    
    # Calculate maximum likelihood
    features.append(calculate_maximum_likelihood(sequence, reference_sequences))
    
    # Calculate Bayesian posterior
    features.append(calculate_bayesian_posterior(sequence, reference_sequences))
    
    # Calculate phylogenetic signal
    features.append(calculate_phylogenetic_signal(sequence))
    
    # Analyze k-mer co-occurrence
    cooccurrence = analyze_kmer_cooccurrence(sequence)
    features.append(len(cooccurrence))  # Pattern richness
    features.append(np.mean(list(cooccurrence.values())))  # Mean co-occurrence
    
    # Calculate positional bias
    bias_scores = calculate_positional_bias(sequence)
    features.append(np.mean(list(bias_scores.values())))  # Mean positional bias
    features.append(np.max(list(bias_scores.values())))   # Max positional bias
    
    return features

def get_phylogenetic_signal_scores(sequence, reference_sequences=None):
    """Calculate comprehensive phylogenetic signal scores.
    
    Args:
        sequence (str): Input sequence
        reference_sequences (list): Optional reference sequences
        
    Returns:
        list: Signal scores including:
            - Blomberg's K
            - Pagel's lambda
            - Moran's I
            - Abouheif's Cmean
            - Phylogenetic eigen vector maps (PEM)
            - D-statistic
            - Phylogenetic signal dissimilarity (PSD)
    """
    scores = []
    
    try:
        # Generate tree and alignment if references not provided
        if reference_sequences is None:
            references = list(get_hamming_neighbors(sequence, max_dist=3))
            references = [r for r in references if is_viable_sequence(r)]
            
        if len(references) < 4:
            return [0.0] * 7  # Return default scores if too few sequences
            
        # Create alignment and tree
        alignment = MultipleSeqAlignment([
            SeqRecord(Seq(seq), id=f"seq_{i}") 
            for i, seq in enumerate([sequence] + references)
        ])
        
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(alignment)
        constructor = DistanceTreeConstructor()
        tree = constructor.build_tree(alignment)
        
        # Calculate Blomberg's K
        k_stat = calculate_blomberg_k(tree, alignment)
        scores.append(k_stat)
        
        # Calculate Pagel's lambda
        lambda_stat = calculate_pagel_lambda(tree, alignment)
        scores.append(lambda_stat)
        
        # Calculate Moran's I
        moran_i = calculate_moran_i(tree, alignment)
        scores.append(moran_i)
        
        # Calculate Abouheif's Cmean
        cmean = calculate_abouheif_cmean(tree, alignment)
        scores.append(cmean)
        
        # Calculate PEM
        pem_score = calculate_pem(tree, alignment)
        scores.append(pem_score)
        
        # Calculate D-statistic
        d_stat = calculate_d_statistic(tree, alignment)
        scores.append(d_stat)
        
        # Calculate PSD
        psd = calculate_psd(tree, alignment)
        scores.append(psd)
        
        return scores
        
    except Exception as e:
        logger.error(f"Error calculating phylogenetic signals: {e}")
        return [0.0] * 7

def calculate_pagel_lambda(tree, alignment):
    """Calculate Pagel's lambda (phylogenetic signal measure)."""
    try:
        # Convert alignment to trait data
        traits = []
        for record in alignment:
            # Use GC content as trait
            gc = (str(record.seq).count('G') + str(record.seq).count('C')) / len(record.seq)
            traits.append(gc)
            
        # Calculate tree variance-covariance matrix
        vcv = calculate_phylo_vcv(tree)
        
        # Find maximum likelihood estimate of lambda
        def neg_log_likelihood(lambda_param):
            modified_vcv = vcv * lambda_param
            np.fill_diagonal(modified_vcv, 1.0)
            try:
                return -np.log(multivariate_normal.pdf(traits, mean=np.mean(traits), cov=modified_vcv))
            except:
                return float('inf')
        
        # Optimize lambda
        result = minimize_scalar(neg_log_likelihood, bounds=(0, 1), method='bounded')
        return result.x
        
    except Exception as e:
        logger.error(f"Error calculating Pagel's lambda: {e}")
        return 0.0

def calculate_moran_i(tree, alignment):
    """Calculate Moran's I spatial autocorrelation coefficient."""
    try:
        # Convert sequences to numeric traits
        traits = []
        for record in alignment:
            gc = (str(record.seq).count('G') + str(record.seq).count('C')) / len(record.seq)
            traits.append(gc)
        
        # Calculate pairwise distances from tree
        distances = calculate_pairwise_distances(tree)
        
        # Calculate Moran's I
        n = len(traits)
        mean_trait = np.mean(traits)
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    weight = 1.0 / (distances[i,j] + 1e-10)  # Avoid division by zero
                    numerator += weight * (traits[i] - mean_trait) * (traits[j] - mean_trait)
                    denominator += weight
                    
        trait_var = np.var(traits)
        if trait_var == 0:
            return 0.0
            
        moran_i = (n / denominator) * (numerator / sum((t - mean_trait)**2 for t in traits))
        return max(-1.0, min(1.0, moran_i))  # Bound between -1 and 1
        
    except Exception as e:
        logger.error(f"Error calculating Moran's I: {e}")
        return 0.0

def calculate_abouheif_cmean(tree, alignment):
    """Calculate Abouheif's Cmean phylogenetic autocorrelation."""
    try:
        # Convert sequences to traits
        traits = []
        for record in alignment:
            gc = (str(record.seq).count('G') + str(record.seq).count('C')) / len(record.seq)
            traits.append(gc)
        
        # Calculate proximity matrix
        W = calculate_phylo_proximity(tree)
        
        # Standardize traits
        traits = np.array(traits)
        traits_std = (traits - np.mean(traits)) / np.std(traits)
        
        # Calculate Cmean
        n = len(traits)
        numerator = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    numerator += W[i,j] * traits_std[i] * traits_std[j]
                    
        denominator = 2 * np.sum(W) / n
        
        if denominator == 0:
            return 0.0
            
        cmean = numerator / denominator
        return max(-1.0, min(1.0, cmean))
        
    except Exception as e:
        logger.error(f"Error calculating Abouheif's Cmean: {e}")
        return 0.0

def calculate_pem(tree, alignment):
    """Calculate Phylogenetic Eigenvector Maps score."""
    try:
        # Get distance matrix
        distances = calculate_pairwise_distances(tree)
        
        # Perform eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(distances)
        
        # Sort eigenvalues and vectors
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]
        
        # Calculate traits
        traits = []
        for record in alignment:
            gc = (str(record.seq).count('G') + str(record.seq).count('C')) / len(record.seq)
            traits.append(gc)
        
        # Calculate PEM score using first few eigenvectors
        n_vectors = min(3, len(eigvals))
        pem_vectors = eigvecs[:, :n_vectors]
        
        # Regression against traits
        reg = LinearRegression()
        reg.fit(pem_vectors, traits)
        
        # Use R² as PEM score
        score = reg.score(pem_vectors, traits)
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        logger.error(f"Error calculating PEM: {e}")
        return 0.0

def calculate_d_statistic(tree, alignment):
    """Calculate Fritz and Purvis' D statistic."""
    try:
        # Convert sequences to binary traits (above/below mean GC)
        traits = []
        gc_contents = []
        for record in alignment:
            gc = (str(record.seq).count('G') + str(record.seq).count('C')) / len(record.seq)
            gc_contents.append(gc)
        
        mean_gc = np.mean(gc_contents)
        traits = [1 if gc > mean_gc else 0 for gc in gc_contents]
        
        # Calculate observed sum of sister-clade differences
        obs_diff = calculate_sister_differences(tree, traits)
        
        # Calculate expected values under Brownian and random models
        brown_diff = simulate_brownian_differences(tree, len(traits))
        rand_diff = simulate_random_differences(tree, len(traits))
        
        # Calculate D statistic
        if brown_diff == rand_diff:
            return 0.0
            
        d_stat = (obs_diff - brown_diff) / (rand_diff - brown_diff)
        return d_stat
        
    except Exception as e:
        logger.error(f"Error calculating D statistic: {e}")
        return 0.0

def calculate_psd(tree, alignment):
    """Calculate Phylogenetic Signal Dissimilarity."""
    try:
        # Calculate pairwise sequence dissimilarities
        seq_dists = calculate_sequence_distances(alignment)
        
        # Calculate phylogenetic distances
        phylo_dists = calculate_pairwise_distances(tree)
        
        # Calculate correlation between distances
        correlation = spearmanr(seq_dists.flatten(), phylo_dists.flatten())[0]
        
        # Convert to PSD score
        psd = (1 + correlation) / 2  # Scale to [0,1]
        return max(0.0, min(1.0, psd))
        
    except Exception as e:
        logger.error(f"Error calculating PSD: {e}")
        return 0.0

# Update create_feature_vector to include new phylogenetic signals
def create_feature_vector(sequence, include_clustering=True, reference_sequences=None):
    """Create comprehensive feature vector including phylogenetic signals."""
    features = []
    
    # ...existing feature extraction code...
    
    # Add phylogenetic signal scores
    features.extend(get_phylogenetic_signal_scores(sequence, reference_sequences))
    
    return features

# Helper functions for phylogenetic calculations
def calculate_phylo_vcv(tree):
    """Calculate phylogenetic variance-covariance matrix."""
    n_tips = len(tree.get_terminals())
    vcv = np.zeros((n_tips, n_tips))
    
    for i, tip1 in enumerate(tree.get_terminals()):
        for j, tip2 in enumerate(tree.get_terminals()):
            if i == j:
                vcv[i,j] = tree.distance(tip1)
            else:
                mrca = tree.common_ancestor(tip1, tip2)
                vcv[i,j] = tree.distance(mrca)
    
    return vcv

def calculate_phylo_proximity(tree):
    """Calculate phylogenetic proximity matrix."""
    n_tips = len(tree.get_terminals())
    W = np.zeros((n_tips, n_tips))
    
    for i, tip1 in enumerate(tree.get_terminals()):
        for j, tip2 in enumerate(tree.get_terminals()):
            if i != j:
                mrca = tree.common_ancestor(tip1, tip2)
                W[i,j] = 1.0 / tree.distance(mrca)
    
    return W

def calculate_sister_differences(tree, traits):
    """Calculate sum of differences between sister clades."""
    diff_sum = 0
    for node in tree.get_nonterminals():
        children = node.clades
        if len(children) == 2:
            clade1_traits = [traits[tree.get_terminals().index(leaf)] 
                           for leaf in children[0].get_terminals()]
            clade2_traits = [traits[tree.get_terminals().index(leaf)] 
                           for leaf in children[1].get_terminals()]
            diff_sum += abs(np.mean(clade1_traits) - np.mean(clade2_traits))
    return diff_sum

def simulate_brownian_differences(tree, n_traits, n_sim=1000):
    """Simulate trait differences under Brownian motion."""
    diffs = []
    for _ in range(n_sim):
        # Simulate Brownian traits
        traits = np.random.normal(0, 1, n_traits)
        diffs.append(calculate_sister_differences(tree, traits))
    return np.mean(diffs)

def simulate_random_differences(tree, n_traits, n_sim=1000):
    """Simulate trait differences under random model."""
    diffs = []
    for _ in range(n_sim):
        # Randomize traits
        traits = np.random.permutation(n_traits)
        diffs.append(calculate_sister_differences(tree, traits))
    return np.mean(diffs)

def calculate_sequence_distances(alignment):
    """Calculate pairwise sequence distances."""
    n_seqs = len(alignment)
    distances = np.zeros((n_seqs, n_seqs))
    
    for i in range(n_seqs):
        for j in range(i+1, n_seqs):
            dist = calculate_sequence_distance(str(alignment[i].seq), 
                                            str(alignment[j].seq))
            distances[i,j] = distances[j,i] = dist
    
    return distances

def calculate_nucleotide_entropy(sequence):
    """Calculate Shannon entropy of nucleotide distributions within k-mer.
    
    Args:
        sequence (str): Input DNA sequence
        
    Returns:
        dict: Various entropy metrics:
            - position_entropy: Entropy at each position
            - dinucleotide_entropy: Entropy of dinucleotide transitions
            - trinucleotide_entropy: Entropy of trinucleotide patterns
            - overall_entropy: Total sequence entropy
    """
    # Calculate positional entropy
    pos_distributions = defaultdict(Counter)
    for i, base in enumerate(sequence):
        pos_distributions[i][base] += 1
    
    pos_entropies = []
    for pos in range(len(sequence)):
        counts = [pos_distributions[pos][base] for base in 'ACGT']
        probs = np.array(counts) / sum(counts)
        pos_entropies.append(entropy(probs, base=2))
    
    # Calculate dinucleotide transition entropy
    dinuc_counts = Counter(sequence[i:i+2] for i in range(len(sequence)-1))
    dinuc_total = sum(dinuc_counts.values())
    dinuc_probs = np.array([count/dinuc_total for count in dinuc_counts.values()])
    dinuc_entropy = entropy(dinuc_probs, base=2)
    
    # Calculate trinucleotide pattern entropy
    trinuc_counts = Counter(sequence[i:i+3] for i in range(len(sequence)-2))
    trinuc_total = sum(trinuc_counts.values())
    trinuc_probs = np.array([count/trinuc_total for count in trinuc_counts.values()])
    trinuc_entropy = entropy(trinuc_probs, base=2)
    
    # Calculate overall sequence entropy
    base_counts = Counter(sequence)
    base_total = sum(base_counts.values())
    base_probs = np.array([base_counts[base]/base_total for base in 'ACGT'])
    overall_entropy = entropy(base_probs, base=2)
    
    return {
        'position_entropy': np.mean(pos_entropies),
        'position_entropy_std': np.std(pos_entropies),
        'dinucleotide_entropy': dinuc_entropy,
        'trinucleotide_entropy': trinuc_entropy,
        'overall_entropy': overall_entropy
    }

def calculate_diversity_metrics(sequence, k=3):
    """Calculate diversity metrics for k-mer composition.
    
    Args:
        sequence (str): Input DNA sequence
        k (int): K-mer size for analysis
        
    Returns:
        dict: Diversity metrics including:
            - simpson_diversity: Simpson's diversity index
            - shannon_diversity: Shannon diversity index
            - evenness: Pielou's evenness index
            - richness: K-mer richness
            - berger_parker: Berger-Parker dominance index
    """
    # Get k-mer counts
    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    counts = Counter(kmers)
    total = sum(counts.values())
    
    # Calculate proportions
    proportions = np.array([count/total for count in counts.values()])
    
    # Simpson's diversity index (1 - D)
    simpson = 1 - sum(p*p for p in proportions)
    
    # Shannon diversity
    shannon = entropy(proportions, base=2)
    
    # Pielou's evenness (normalized Shannon)
    max_shannon = np.log2(len(counts))  # Maximum possible Shannon entropy
    evenness = shannon / max_shannon if max_shannon > 0 else 0
    
    # K-mer richness (number of unique k-mers)
    richness = len(counts)
    
    # Berger-Parker dominance (proportion of most abundant k-mer)
    berger_parker = max(proportions)
    
    return {
        'simpson_diversity': simpson,
        'shannon_diversity': shannon,
        'evenness': evenness,
        'richness': richness,
        'berger_parker': berger_parker
    }

def analyze_overlapping_constraints(sequence):
    """Analyze sequence constraints from overlapping functional elements.
    
    Args:
        sequence (str): Input DNA sequence
        
    Returns:
        dict: Constraint scores including:
            - overlap_density: Density of overlapping elements
            - frame_conservation: Conservation across reading frames
            - regulatory_density: Density of regulatory motifs
            - structure_constraints: RNA structural constraints
    """
    constraints = {
        'overlap_density': 0.0,
        'frame_conservation': 0.0,
        'regulatory_density': 0.0,
        'structure_constraints': 0.0
    }
    
    # Analyze overlapping coding potential in different frames
    frames = []
    for frame in range(3):
        codons = [sequence[i:i+3] for i in range(frame, len(sequence)-2, 3)]
        frames.append(analyze_coding_potential(codons))
    
    # Calculate frame conservation score
    constraints['frame_conservation'] = np.mean([
        frames[i]['coding_score'] for i in range(len(frames))
    ])
    
    # Find overlapping functional elements
    elements = find_functional_elements(sequence)
    overlap_regions = find_overlapping_regions(elements)
    constraints['overlap_density'] = len(overlap_regions) / len(sequence)
    
    # Score regulatory motif density
    reg_motifs = find_regulatory_motifs(sequence)
    constraints['regulatory_density'] = sum(len(m) for m in reg_motifs) / len(sequence)
    
    # Analyze RNA structural constraints
    constraints['structure_constraints'] = analyze_rna_constraints(sequence)
    
    return constraints

def analyze_coding_potential(codons):
    """Analyze coding potential of codon sequence."""
    standard_table = CodonTable.standard_dna_table
    
    # Calculate codon usage frequencies
    codon_freq = Counter(codons)
    total_codons = len(codons)
    
    # Score coding potential based on:
    # 1. Start/stop codon positioning
    # 2. Codon bias relative to known genes
    # 3. Amino acid property conservation
    
    score = {
        'coding_score': 0.0,
        'start_codons': [],
        'stop_codons': [],
        'biased_codons': defaultdict(float)
    }
    
    # Check start/stop codons
    if codons and codons[0] in standard_table.start_codons:
        score['coding_score'] += 0.2
        score['start_codons'].append(0)
    
    for i, codon in enumerate(codons):
        if codon in standard_table.stop_codons:
            score['stop_codons'].append(i)
            if i == len(codons) - 1:  # Well-positioned stop codon
                score['coding_score'] += 0.2
    
    # Analyze codon bias
    for codon, freq in codon_freq.items():
        if codon in standard_table.forward_table:
            rel_freq = freq / total_codons
            score['biased_codons'][codon] = rel_freq
            
            # Compare to expected frequencies
            expected = 1.0 / len(standard_table.forward_table)
            if abs(rel_freq - expected) > 0.1:  # Significant bias
                score['coding_score'] += 0.1
    
    return score

def find_functional_elements(sequence):
    """Identify potential functional sequence elements."""
    elements = []
    
    # Known regulatory motifs
    regulatory_patterns = {
        'shine_dalgarno': r'AGGAGG',
        'kozak': r'GCC[AG]CCATGG',
        'tata_box': r'TATA[AT]A',
        'promoter': r'TTGACA.{15,19}TATAAT',
        'splice_donor': r'GT[AG]AGT',
        'splice_acceptor': r'[CT]AG',
        'polya_signal': r'AATAAA'
    }
    
    # Search for regulatory elements
    for element_type, pattern in regulatory_patterns.items():
        for match in re.finditer(pattern, sequence):
            elements.append({
                'type': element_type,
                'start': match.start(),
                'end': match.end(),
                'sequence': match.group()
            })
    
    # Find potential coding regions
    orfs = find_orfs(sequence)
    for orf in orfs:
        elements.append({
            'type': 'coding',
            'start': orf['start'],
            'end': orf['end'],
            'sequence': orf['sequence']
        })
    
    return elements

def find_overlapping_regions(elements):
    """Find regions with overlapping functional elements."""
    overlaps = []
    
    # Sort elements by start position
    sorted_elements = sorted(elements, key=lambda x: x['start'])
    
    # Find overlaps
    for i in range(len(sorted_elements)-1):
        current = sorted_elements[i]
        next_elem = sorted_elements[i+1]
        
        if current['end'] > next_elem['start']:
            overlaps.append({
                'elements': [current, next_elem],
                'start': next_elem['start'],
                'end': min(current['end'], next_elem['end']),
                'length': min(current['end'], next_elem['end']) - next_elem['start']
            })
    
    return overlaps

def find_regulatory_motifs(sequence):
    """Find known regulatory motifs and binding sites."""
    motifs = []
    
    # Common regulatory motifs with their consensus sequences
    regulatory_motifs = {
        'ribosome_binding': {
            'pattern': r'AGGAGG',
            'score': 1.0
        },
        'kozak_context': {
            'pattern': r'GCC[AG]CCATGG',
            'score': 1.0
        },
        'promoter_elements': {
            'pattern': r'TTGACA|TATAAT',
            'score': 0.8
        },
        'transcription_factor': {
            'pattern': r'TGACTCA|CCCGCC|CACGTG',
            'score': 0.7
        },
        'splicing_signals': {
            'pattern': r'GT[AG]AGT|[CT]AG',
            'score': 0.9
        }
    }
    
    for motif_type, data in regulatory_motifs.items():
        for match in re.finditer(data['pattern'], sequence):
            motifs.append({
                'type': motif_type,
                'position': match.start(),
                'sequence': match.group(),
                'score': data['score']
            })
    
    return motifs

def analyze_rna_constraints(sequence):
    """Analyze potential RNA structural constraints."""
    # Simple prediction of RNA structural potential
    pairs = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    
    # Look for potential stem-loop structures
    stem_loops = []
    min_stem = 4
    max_loop = 8
    
    for i in range(len(sequence)-min_stem*2):
        for stem_len in range(min_stem, min(8, (len(sequence)-i)//2)):
            for loop_len in range(3, max_loop+1):
                if i + stem_len*2 + loop_len > len(sequence):
                    continue
                    
                left_stem = sequence[i:i+stem_len]
                loop = sequence[i+stem_len:i+stem_len+loop_len]
                right_stem = sequence[i+stem_len+loop_len:i+stem_len*2+loop_len]
                
                # Check complementarity
                matches = sum(1 for x, y in zip(left_stem, right_stem[::-1])
                            if pairs.get(x) == y)
                
                if matches >= stem_len * 0.75:  # Allow some mismatches
                    stem_loops.append({
                        'start': i,
                        'end': i+stem_len*2+loop_len,
                        'score': matches/stem_len
                    })
    
    # Calculate constraint score based on structural elements
    if not stem_loops:
        return 0.0
        
    return np.mean([s['score'] for s in stem_loops])

def find_orfs(sequence, min_length=30):
    """Find potential open reading frames."""
    orfs = []
    start_codons = {'ATG', 'GTG', 'TTG'}
    stop_codons = {'TAA', 'TAG', 'TGA'}
    
    # Search in all frames
    for frame in range(3):
        index = frame
        while index < len(sequence)-2:
            codon = sequence[index:index+3]
            
            if codon in start_codons:
                # Found start codon, look for stop
                orf_start = index
                orf_seq = []
                
                while index < len(sequence)-2:
                    codon = sequence[index:index+3]
                    orf_seq.append(codon)
                    
                    if codon in stop_codons:
                        if len(orf_seq)*3 >= min_length:
                            orfs.append({
                                'start': orf_start,
                                'end': index + 3,
                                'frame': frame,
                                'sequence': ''.join(orf_seq)
                            })
                        break
                        
                    index += 3
            index += 3
    
    return orfs

# Update create_feature_vector to include constraint analysis
def create_feature_vector(sequence, include_clustering=True, reference_sequences=None):
    """Create comprehensive feature vector including constraint analysis."""
    features = []
    
    # Add existing features
    # ...existing feature extraction code...
    
    # Add constraint analysis features
    constraints = analyze_overlapping_constraints(sequence)
    features.extend([
        constraints['overlap_density'],
        constraints['frame_conservation'],
        constraints['regulatory_density'],
        constraints['structure_constraints']
    ])
    
    # Add motif scores
    motifs = find_regulatory_motifs(sequence)
    motif_score = sum(m['score'] for m in motifs) / len(sequence) if motifs else 0
    features.append(motif_score)
    
    # Add spectral analysis features
    features.extend(get_spectral_features(sequence))
    
    return features

def sequence_to_signal(sequence):
    """Convert DNA sequence to numerical signal.
    
    Args:
        sequence (str): DNA sequence
        
    Returns:
        tuple: (complex signal array, nucleotide mapping)
    """
    # Complex number mapping for nucleotides (tetrahedron vertices)
    mapping = {
        'A': 1 + 1j,
        'T': -1 - 1j,
        'G': -1 + 1j,
        'C': 1 - 1j
    }
    return np.array([mapping[base] for base in sequence]), mapping

def analyze_sequence_periodicity(sequence):
    """Analyze sequence periodicity using Fourier transform.
    
    Args:
        sequence (str): Input DNA sequence
        
    Returns:
        dict: Spectral features including:
            - dominant_periods: List of dominant periodic components
            - spectral_entropy: Entropy of frequency distribution
            - period_strengths: Strength of key periodic signals
    """
    # Convert sequence to signal
    signal_array, _ = sequence_to_signal(sequence)
    
    # Compute FFT
    n = len(signal_array)
    freqs = fftfreq(n)
    fft_vals = fft(signal_array)
    
    # Get power spectrum
    power_spectrum = np.abs(fft_vals)**2
    normalized_spectrum = power_spectrum / np.sum(power_spectrum)
    
    # Find dominant periods
    peaks = signal.find_peaks(power_spectrum[:n//2])[0]
    periods = 1/freqs[peaks]
    peak_powers = power_spectrum[peaks]
    
    # Sort by power and get top periods
    sorted_indices = np.argsort(peak_powers)[::-1]
    dominant_periods = periods[sorted_indices][:5]  # Top 5 periods
    
    # Calculate spectral entropy
    spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))
    
    # Calculate strength of key periodic components
    period_strengths = {
        '3bp': np.mean(power_spectrum[freqs == 1/3]),  # Codon periodicity
        '10bp': np.mean(power_spectrum[freqs == 1/10]),  # DNA helical pitch
        '200bp': np.mean(power_spectrum[freqs == 1/200])  # Nucleosome spacing
    }
    
    return {
        'dominant_periods': dominant_periods,
        'spectral_entropy': spectral_entropy,
        'period_strengths': period_strengths
    }

def wavelet_decomposition(sequence, wavelet='db1', level=5):
    """Perform wavelet decomposition of sequence.
    
    Args:
        sequence (str): Input DNA sequence
        wavelet (str): Wavelet type to use
        level (int): Decomposition level
        
    Returns:
        dict: Wavelet features including:
            - coefficients: Wavelet coefficients at each level
            - energy: Energy distribution across scales
            - entropy: Entropy at each decomposition level
    """
    # Convert sequence to signal
    signal_array, _ = sequence_to_signal(sequence)
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal_array.real, wavelet, level=level)
    
    # Calculate energy at each level
    energy = [np.sum(np.abs(c)**2) for c in coeffs]
    total_energy = sum(energy)
    energy_dist = [e/total_energy for e in energy]
    
    # Calculate entropy at each level
    entropy = []
    for c in coeffs:
        pdf = np.abs(c)**2 / np.sum(np.abs(c)**2)
        entropy.append(-np.sum(pdf * np.log2(pdf + 1e-10)))
    
    return {
        'coefficients': coeffs,
        'energy': energy_dist,
        'entropy': entropy
    }

def get_spectral_features(sequence):
    """Extract spectral and wavelet-based features from sequence.
    
    Args:
        sequence (str): Input DNA sequence
        
    Returns:
        list: Features including:
            - Dominant period strengths
            - Spectral entropy
            - Wavelet energy distribution
            - Wavelet entropy profile
    """
    features = []
    
    # Get Fourier-based periodicity features
    periodicity = analyze_sequence_periodicity(sequence)
    features.extend([
        np.mean(periodicity['dominant_periods']),
        periodicity['spectral_entropy'],
        periodicity['period_strengths']['3bp'],
        periodicity['period_strengths']['10bp'],
        periodicity['period_strengths']['200bp']
    ])
    
    # Get wavelet decomposition features
    wavelet_features = wavelet_decomposition(sequence)
    features.extend([
        np.mean(wavelet_features['energy']),  # Mean energy distribution
        np.std(wavelet_features['energy']),   # Energy variability
        np.mean(wavelet_features['entropy']), # Mean entropy across scales
        max(wavelet_features['entropy']),     # Maximum scale entropy
        min(wavelet_features['entropy'])      # Minimum scale entropy
    ])
    
    return features

class ScalingAnalysis:
    """Analyze how biological sequence clustering behavior scales with dataset size."""
    
    def __init__(self, initial_size=1000, max_size=100000, scale_factor=2):
        self.initial_size = initial_size
        self.max_size = max_size
        self.scale_factor = scale_factor
        self.results = defaultdict(list)
        
    def generate_dataset(self, size, k_size):
        """Generate balanced dataset of biological and random k-mers."""
        # Extract biological k-mers
        bio_kmers = []
        fasta_path = config_manager.get('PATHS', 'databases', 'biological_sequences')
        
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq)
            kmers = [seq[i:i+k_size] for i in range(len(seq)-k_size+1)]
            bio_kmers.extend(kmers[:size//2])  # Take half the requested size
            
            if len(bio_kmers) >= size//2:
                break
        
        # Generate random k-mers
        random_kmers = generate_random_kmers(size//2, k_size)
        
        # Create feature vectors
        X_bio = np.array([create_feature_vector(kmer) for kmer in bio_kmers])
        X_random = np.array([create_feature_vector(kmer) for kmer in random_kmers])
        
        X = np.vstack([X_bio, X_random])
        y = np.hstack([np.ones(len(X_bio)), np.zeros(len(X_random))])
        
        return X, y
    
    def calculate_clustering_metrics(self, X, y):
        """Calculate metrics for clustering quality."""
        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
        cluster_labels = clusterer.fit_predict(X)
        
        metrics = {
            'silhouette': silhouette_score(X, y) if len(np.unique(y)) > 1 else 0,
            'calinski': calinski_harabasz_score(X, y),
            'hdbscan_score': np.mean(clusterer.outlier_scores_),
            'cluster_persistence': np.mean(clusterer.cluster_persistence_)
        }
        
        return metrics
    
    def analyze_kmer_size(self, k_size):
        """Analyze scaling behavior for specific k-mer size."""
        current_size = self.initial_size
        
        while current_size <= self.max_size:
            # Generate dataset
            X, y = self.generate_dataset(current_size, k_size)
            
            # Calculate metrics
            metrics = self.calculate_clustering_metrics(X, y)
            
            # Store results
            self.results[k_size].append({
                'size': current_size,
                'metrics': metrics,
                'sample_X': X[:1000],  # Store subset for visualization
                'sample_y': y[:1000]
            })
            
            # Check for tapering
            if self.check_tapering(k_size):
                break
                
            current_size = int(current_size * self.scale_factor)
    
    def check_tapering(self, k_size, window=3):
        """Check if clustering metrics have tapered off."""
        if len(self.results[k_size]) < window:
            return False
            
        recent_scores = [r['metrics']['silhouette'] 
                        for r in self.results[k_size][-window:]]
        
        # Calculate rate of change
        changes = np.diff(recent_scores)
        mean_change = np.mean(np.abs(changes))
        
        return mean_change < 0.01  # Threshold for tapering
    
    def visualize_scaling(self, k_size, method='tsne'):
        """Visualize clustering at different scales."""
        results = self.results[k_size]
        n_plots = len(results)
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        for ax, result in zip(axes, results):
            X, y = result['sample_X'], result['sample_y']
            
            # Dimensionality reduction
            if method == 'tsne':
                X_2d = TSNE(n_components=2).fit_transform(X)
            elif method == 'umap':
                X_2d = umap.UMAP().fit_transform(X)
            else:
                X_2d = PCA(n_components=2).fit_transform(X)
            
            # Plot clusters
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, 
                               cmap='viridis', alpha=0.6)
            ax.set_title(f'N={result["size"]}\nSilhouette={result["metrics"]["silhouette"]:.3f}')
            
        plt.colorbar(scatter, ax=axes[-1], label='Class')
        plt.suptitle(f'Clustering Evolution for {k_size}-mers')
        plt.tight_layout()
        plt.show()
    
    def plot_metric_trends(self):
        """Plot how clustering metrics change with scale."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        metrics = ['silhouette', 'calinski', 'hdbscan_score', 'cluster_persistence']
        
        for ax, metric in zip(axes, metrics):
            for k_size in self.results.keys():
                sizes = [r['size'] for r in self.results[k_size]]
                scores = [r['metrics'][metric] for r in self.results[k_size]]
                
                ax.plot(sizes, scores, 'o-', label=f'{k_size}-mers')
                
            ax.set_xlabel('Dataset Size')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def analyze_kmer_scaling(k_sizes=[31, 33, 35], initial_size=1000, max_size=100000):
    """Run complete scaling analysis for multiple k-mer sizes."""
    analyzer = ScalingAnalysis(initial_size, max_size)
    
    for k_size in k_sizes:
        print(f"\nAnalyzing {k_size}-mers...")
        analyzer.analyze_kmer_size(k_size)
        analyzer.visualize_scaling(k_size, method='umap')
    
    analyzer.plot_metric_trends()
    return analyzer.results

if __name__ == "__main__":
    # Example usage
    logger = setup_logging()
    logger.info("Starting k-mer scaling analysis")
    
    try:
        # Run scaling analysis
        results = analyze_kmer_scaling(
            k_sizes=[31, 33, 35],
            initial_size=1000,
            max_size=50000
        )
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"scaling_analysis_{timestamp}.joblib"
        dump(results, output_file)
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error("Analysis failed", exc_info=True)
        raise

class BiologicalManifold:
    """Representation of biological sequence space as a manifold."""
    
    def __init__(self, dimension=10):
        self.dimension = dimension
        self.graph = nx.Graph()
        self.embedder = Isomap(n_components=dimension, n_neighbors=15)
        self.neutral_regions = []
        self.conserved_regions = []
        
    def fit_manifold(self, sequences, labels=None):
        """Learn manifold structure from sequences."""
        # Calculate pairwise distances
        distances = np.zeros((len(sequences), len(sequences)))
        for i, seq1 in enumerate(sequences):
            for j, seq2 in enumerate(sequences[i:], i):
                dist = calculate_sequence_distance(seq1, seq2)
                distances[i,j] = distances[j,i] = dist
        
        # Learn manifold structure
        self.embedding = self.embedder.fit_transform(distances)
        
        # Build neighborhood graph
        for i in range(len(sequences)):
            for j in range(i+1, len(sequences)):
                if distances[i,j] < 0.3:  # Connection threshold
                    self.graph.add_edge(i, j, weight=distances[i,j])
        
        # Identify conserved and neutral regions
        self._identify_regions(sequences, distances)
        
    def _identify_regions(self, sequences, distances):
        """Identify conserved and neutral regions in the manifold."""
        # Find dense regions (conserved)
        components = list(nx.connected_components(self.graph))
        for component in components:
            if len(component) > 5:  # Minimum size threshold
                subgraph = self.graph.subgraph(component)
                density = nx.density(subgraph)
                if (density > 0.7):
                    self.conserved_regions.append(component)
                elif 0.3 <= density <= 0.7:
                    self.neutral_regions.append(component)
    
    def get_nearest_neighbors(self, sequence, k=5):
        """Find nearest neighbors in manifold space."""
        # Convert sequence to feature vector
        features = create_feature_vector(sequence)
        
        # Project onto manifold
        projection = self.embedder.transform([features])
        
        # Find k nearest neighbors
        distances = np.linalg.norm(self.embedding - projection, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        
        return nearest_indices, distances[nearest_indices]
    
    def is_in_conserved_region(self, sequence):
        """Check if sequence lies in a conserved region."""
        neighbors, distances = self.get_nearest_neighbors(sequence, k=3)
        return np.mean(distances) < 0.2  # Conservative threshold
    
    def get_neutral_drift_path(self, start_seq, end_seq, max_steps=100):
        """Find path through neutral regions between sequences."""
        start_neighbors = self.get_nearest_neighbors(start_seq)[0]
        end_neighbors = self.get_nearest_neighbors(end_seq)[0]
        
        # Find path through graph
        try:
            path = nx.shortest_path(self.graph, 
                                  source=start_neighbors[0],
                                  target=end_neighbors[0],
                                  weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None

class ManifoldGuidedExplorer:
    """Explore sequence space using manifold structure."""
    
    def __init__(self, manifold, mutation_rate=0.01):
        self.manifold = manifold
        self.mutation_rate = mutation_rate
        self.current_position = None
        self.history = []
        
    def initialize(self, sequence):
        """Initialize explorer at a sequence position."""
        self.current_position = sequence
        self.history.append(sequence)
        
    def step(self):
        """Take one step through the manifold."""
        if self.current_position is None:
            raise ValueError("Explorer not initialized")
            
        # Get current neighborhood
        neighbors, distances = self.manifold.get_nearest_neighbors(
            self.current_position)
            
        # Decide movement strategy
        if self.manifold.is_in_conserved_region(self.current_position):
            # Make conservative moves in conserved regions
            step = self._conservative_step(neighbors, distances)
        else:
            # Allow more exploration in neutral regions
            step = self._exploratory_step(neighbors, distances)
            
        self.current_position = step
        self.history.append(step)
        return step
    
    def _conservative_step(self, neighbors, distances):
        """Take small steps in conserved regions."""
        # Choose closest neighbor with small random perturbation
        weights = 1 / (distances + 1e-6)
        choice = np.random.choice(neighbors, p=weights/np.sum(weights))
        
        # Make minimal mutations
        sequence = self.current_position
        n_mutations = np.random.poisson(self.mutation_rate)
        positions = np.random.choice(len(sequence), size=n_mutations)
        
        for pos in positions:
            options = [b for b in 'ACGT' if b != sequence[pos]]
            sequence = sequence[:pos] + np.random.choice(options) + sequence[pos+1:]
            
        return sequence
    
    def _exploratory_step(self, neighbors, distances):
        """Take larger steps in neutral regions."""
        # Allow jumping to more distant neighbors
        weights = np.ones_like(distances) / len(distances)
        choice = np.random.choice(neighbors, p=weights)
        
        # Make more mutations
        sequence = self.current_position
        n_mutations = np.random.poisson(self.mutation_rate * 3)
        positions = np.random.choice(len(sequence), size=n_mutations)
        
        for pos in positions:
            sequence = sequence[:pos] + np.random.choice('ACGT') + sequence[pos+1:]
            
        return sequence

def analyze_manifold_structure(sequences, k_size=31):
    """Analyze the structure of the biological sequence manifold."""
    # Create manifold
    manifold = BiologicalManifold(dimension=min(10, len(sequences)))
    manifold.fit_manifold(sequences)
    
    # Analyze connectivity
    connectivity = nx.average_node_connectivity(manifold.graph)
    clustering = nx.average_clustering(manifold.graph)
    
    # Analyze regions
    n_conserved = len(manifold.conserved_regions)
    n_neutral = len(manifold.neutral_regions)
    
    print(f"Manifold Analysis Results:")
    print(f"Connectivity: {connectivity:.3f}")
    print(f"Clustering: {clustering:.3f}")
    print(f"Conserved Regions: {n_conserved}")
    print(f"Neutral Regions: {n_neutral}")
    
    return manifold

# Update create_feature_vector to account for manifold structure
def create_feature_vector(sequence, manifold=None):
    """Create feature vector considering manifold position."""
    features = []
    
    # Add basic features
    # ...existing feature extraction code...
    
    # Add manifold-aware features if available
    if manifold is not None:
        neighbors, distances = manifold.get_nearest_neighbors(sequence)
        features.extend([
            np.mean(distances),  # Local density
            np.std(distances),   # Neighborhood variability
            1 if manifold.is_in_conserved_region(sequence) else 0
        ])
    
    return features

# Example usage in main
if __name__ == "__main__":
    # Load sequences
    sequences = []
    with open("biological_sequences.fasta", "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            sequences.append(str(record.seq))
    
    # Analyze manifold structure
    manifold = analyze_manifold_structure(sequences)
    
    # Initialize explorer
    explorer = ManifoldGuidedExplorer(manifold)
    explorer.initialize(sequences[0])
    
    # Explore sequence space
    for _ in range(100):
        sequence = explorer.step()
        if manifold.is_in_conserved_region(sequence):
            print("Found conserved region")

# ...existing imports...
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

# ...existing code...

class SequenceVisualizer:
    """Advanced visualization tools for sequence analysis"""
    
    def __init__(self, style='publication'):
        self.style = style
        self.set_style()
        
    def set_style(self):
        """Set publication-quality plotting style"""
        if self.style == 'publication':
            plt.style.use('seaborn-whitegrid')
            sns.set_context("paper", font_scale=1.5)
            plt.rcParams['figure.figsize'] = [12, 8]
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial']
    
    def plot_manifold(self, manifold, sequences, filename=None):
        """Create interactive 3D visualization of sequence manifold"""
        # Get 3D embedding
        embedding = TSNE(n_components=3).fit_transform(manifold.embedding)
        
        # Create interactive 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=manifold.graph.degree(),
                colorscale='Viridis',
                opacity=0.8
            ),
            hovertext=[f"Sequence {i}" for i in range(len(sequences))]
        )])
        
        fig.update_layout(
            title="Biological Sequence Manifold",
            scene=dict(
                xaxis_title="TSNE-1",
                yaxis_title="TSNE-2",
                zaxis_title="TSNE-3"
            ),
            width=1200,
            height=800,
        )
        
        if filename:
            fig.write_html(filename)
        return fig
    
    def plot_feature_importance(self, clf, feature_names, filename=None):
        """Create enhanced feature importance visualization"""
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(12, len(feature_names)*0.3))
        
        # Create horizontal bar plot
        ax = sns.barplot(
            data=importance.tail(20),  # Show top 20 features
            x='Importance',
            y='Feature',
            palette='viridis'
        )
        
        # Enhance styling
        ax.set_title('Top 20 Most Important Features', pad=20, fontsize=16)
        ax.set_xlabel('Relative Importance', fontsize=12)
        ax.set_ylabel('')
        
        # Add value labels
        for i, v in enumerate(importance['Importance'].tail(20)):
            ax.text(v, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
    
    def plot_evolutionary_landscape(self, sequences, scores, filename=None):
        """Create 3D evolutionary fitness landscape visualization"""
        # Calculate pairwise distances
        distances = squareform(pdist([create_feature_vector(s) for s in sequences]))
        
        # Create 2D embedding
        embedding = MDS(n_components=2).fit_transform(distances)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=embedding[:, 0].reshape(-1, 10),
            y=embedding[:, 1].reshape(-1, 10),
            z=np.array(scores).reshape(-1, 10),
            colorscale='Viridis'
        )])
        
        fig.update_layout(
            title='Evolutionary Fitness Landscape',
            scene=dict(
                xaxis_title='Sequence Space Dimension 1',
                yaxis_title='Sequence Space Dimension 2',
                zaxis_title='Fitness Score'
            ),
            width=1000,
            height=800
        )
        
        if filename:
            fig.write_html(filename)
        return fig
    
    def plot_constraint_network(self, elements, filename=None):
        """Visualize network of overlapping sequence constraints"""
        G = nx.Graph()
        
        # Create network from overlapping elements
        for e1 in elements:
            for e2 in elements:
                if e1 != e2:
                    if (e1['end'] > e2['start'] and e1['start'] < e2['end']):
                        overlap = min(e1['end'], e2['end']) - max(e1['start'], e2['start'])
                        if overlap > 0:
                            G.add_edge(e1['type'], e2['type'], weight=overlap)
        
        # Calculate node sizes based on element frequencies
        sizes = [G.degree(node, weight='weight')*100 for node in G.nodes()]
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title("Sequence Constraint Network", pad=20, fontsize=16)
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
    
    def plot_spectral_analysis(self, sequence, filename=None):
        """Create publication-quality spectral analysis visualization"""
        # Get spectral features
        periodicity = analyze_sequence_periodicity(sequence)
        wavelet = wavelet_decomposition(sequence)
        
        # Create subplot figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # Plot power spectrum
        ax1 = fig.add_subplot(gs[0, 0])
        freqs = np.fft.fftfreq(len(sequence))
        power = np.abs(np.fft.fft(sequence_to_signal(sequence)[0]))**2
        ax1.plot(freqs[1:len(freqs)//2], power[1:len(freqs)//2], 'b-', alpha=0.7)
        ax1.set_title('Power Spectrum', fontsize=12)
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Power')
        
        # Plot wavelet scalogram
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(
            np.abs(wavelet['coefficients'][1:]),
            aspect='auto',
            cmap='viridis'
        )
        ax2.set_title('Wavelet Scalogram', fontsize=12)
        ax2.set_ylabel('Scale')
        ax2.set_xlabel('Position')
        plt.colorbar(im, ax=ax2)
        
        # Plot dominant periods
        ax3 = fig.add_subplot(gs[1, 0])
        periods = periodicity['dominant_periods']
        ax3.bar(range(len(periods)), periods, color='purple', alpha=0.6)
        ax3.set_title('Dominant Periods', fontsize=12)
        ax3.set_xlabel('Rank')
        ax3.set_ylabel('Period Length')
        
        # Plot wavelet energy distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(wavelet['energy'], 'r-o', alpha=0.7)
        ax4.set_title('Wavelet Energy Distribution', fontsize=12)
        ax4.set_xlabel('Decomposition Level')
        ax4.set_ylabel('Relative Energy')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()

# Add to main analysis pipeline
def visualize_analysis_results(clf, X_test, y_test, sequences, manifold):
    """Create comprehensive visualization of analysis results"""
    viz = SequenceVisualizer()
    
    # Create manifold visualization
    viz.plot_manifold(manifold, sequences, "manifold_visualization.html")
    
    # Plot feature importance
    viz.plot_feature_importance(clf, get_feature_names(), "feature_importance.png")
    
    # Plot evolutionary landscape
    scores = clf.predict_proba(X_test)[:, 1]
    viz.plot_evolutionary_landscape(sequences, scores, "evolutionary_landscape.html")
    
    # Plot constraint network
    elements = []
    for seq in sequences[:100]:  # Analyze subset for visualization
        elements.extend(find_functional_elements(seq))
    viz.plot_constraint_network(elements, "constraint_network.png")
    
    # Plot spectral analysis for example sequence
    viz.plot_spectral_analysis(sequences[0], "spectral_analysis.png")

# Update main execution
if __name__ == "__main__":
    # ...existing code...
    
    # Add visualization of results
    visualize_analysis_results(clf, X_test, y_test, sequences, manifold)

# Add protein translation handling and k-mer generation from reverse-translated sequences

def reverse_translate_protein(protein_seq):
    """Reverse translate protein sequence to DNA using most common codons."""
    # Most common codons per amino acid (based on human genome)
    codon_table = {
        'A': 'GCC', 'C': 'TGC', 'D': 'GAC', 'E': 'GAG',
        'F': 'TTC', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC',
        'K': 'AAG', 'L': 'CTG', 'M': 'ATG', 'N': 'AAC',
        'P': 'CCC', 'Q': 'CAG', 'R': 'CGG', 'S': 'AGC',
        'T': 'ACC', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAC',
        '*': 'TGA', 'X': 'NNN', 'B': 'GAY', 'Z': 'GAR',
        'J': 'MTY', 'U': 'TGA'  # Special cases
    }
    
    return ''.join(codon_table.get(aa, 'NNN') for aa in protein_seq.upper())

def get_all_codons_for_aa(aa):
    """Get all possible codons for an amino acid.
    
    Args:
        aa (str): Single letter amino acid code
        
    Returns:
        list: All possible codons for the amino acid
    """
    # Standard genetic code with all possible codons
    codon_table = {
        'A': ['GCT', 'GCC', 'GCA', 'GCG'],
        'C': ['TGT', 'TGC'],
        'D': ['GAT', 'GAC'],
        'E': ['GAA', 'GAG'],
        'F': ['TTT', 'TTC'],
        'G': ['GGT', 'GGC', 'GGA', 'GGG'],
        'H': ['CAT', 'CAC'],
        'I': ['ATT', 'ATC', 'ATA'],
        'K': ['AAA', 'AAG'],
        'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
        'M': ['ATG'],
        'N': ['AAT', 'AAC'],
        'P': ['CCT', 'CCC', 'CCA', 'CCG'],
        'Q': ['CAA', 'CAG'],
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'],
        'V': ['GTT', 'GTC', 'GTA', 'GTG'],
        'W': ['TGG'],
        'Y': ['TAT', 'TAC'],
        '*': ['TAA', 'TAG', 'TGA'],
        'X': ['NNN'],
        'B': ['GAT', 'GAC'],  # Aspartic acid or Asparagine
        'Z': ['GAA', 'GAG', 'CAA', 'CAG'],  # Glutamic acid or Glutamine
        'J': ['ATT', 'ATC', 'ATA', 'CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG'],  # Leucine or Isoleucine
        'U': ['TGA']  # Selenocysteine
    }
    return codon_table.get(aa.upper(), ['NNN'])

def generate_all_reverse_translations(protein_seq, max_sequences=10000):
    """Generate all possible DNA sequences for a protein sequence.
    
    Args:
        protein_seq (str): Input protein sequence
        max_sequences (int): Maximum number of sequences to generate
        
    Returns:
        generator: Generator yielding possible DNA sequences
    """
    # Get possible codons for first amino acid
    current_sequences = get_all_codons_for_aa(protein_seq[0])
    
    # Iteratively build sequences for each subsequent amino acid
    for aa in protein_seq[1:]:
        possible_codons = get_all_codons_for_aa(aa)
        new_sequences = []
        
        # Calculate how many sequences we'll have after adding this amino acid
        total_combinations = len(current_sequences) * len(possible_codons)
        
        # If we'll exceed max_sequences, sample randomly
        if total_combinations > max_sequences:
            current_sequences = np.random.choice(current_sequences, 
                                              size=max_sequences//len(possible_codons),
                                              replace=False)
        
        # Add all possible codons to current sequences
        for seq in current_sequences:
            for codon in possible_codons:
                new_sequences.append(seq + codon)
                
                # Yield sequence if we're at max to avoid memory issues
                if len(new_sequences) >= max_sequences:
                    yield from new_sequences
                    new_sequences = []
        
        current_sequences = new_sequences
        
        # Break if we've generated enough sequences
        if len(current_sequences) >= max_sequences:
            break
    
    # Yield any remaining sequences
    yield from current_sequences

def create_kmer_dataset(fasta_file="uniprot_sprot.fasta", phylome_file=None, alignment_file=None, 
                       k_size=config_manager.get('MODEL', 'kmer', 'size'), 
                       negative_sample_size=None, event_tolerance=True, 
                       include_known_elements=True, chunk_size=config_manager.get('MODEL', 'training', 'batch_size'), 
                       hmm_db_path=config_manager.get('PATHS', 'databases', 'hmm'),
                       max_translations_per_protein=1000):
    """Create dataset with streaming data handling using all possible reverse translations."""
    X, y = [], []
    biological_kmers = BloomFilter(1000000, 5)
    
    # Process protein sequences in chunks
    for chunk in sequence_generator(fasta_file, chunk_size):
        chunk_kmers = set()
        
        # Process each protein sequence
        for protein_seq in chunk:
            # Generate all possible DNA sequences
            for dna_seq in generate_all_reverse_translations(protein_seq, 
                                                           max_sequences=max_translations_per_protein):
                if 'N' in dna_seq:
                    continue
                    
                # Generate k-mers from this DNA sequence
                for i in range(len(dna_seq) - k_size + 1):
                    kmer = dna_seq[i:i+k_size]
                    if 'N' not in kmer and kmer not in chunk_kmers:
                        chunk_kmers.add(kmer)
                        biological_kmers.add(kmer)
                        features = create_feature_vector(kmer, hmm_db_path=hmm_db_path)
                        X.append(features)
                        y.append(1)
        
        # Generate negative examples for this chunk
        chunk_size = len(X)
        random_kmers = list(generate_random_kmers(chunk_size, k_size))
        
        for kmer in random_kmers:
            if kmer not in biological_kmers:
                features = create_feature_vector(kmer, hmm_db_path=hmm_db_path)
                X.append(features)
                y.append(0)
        
        # Process in batches
        if len(X) >= chunk_size * 2:
            X_chunk = np.array(X)
            y_chunk = np.array(y)
            yield X_chunk, y_chunk
            X, y = [], []
    
    # Yield remaining data
    if X:
        yield np.array(X), np.array(y)

# Update main execution to use Swiss-Prot
if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting k-mer analysis using reverse-translated UniProt/Swiss-Prot")
    
    try:
        # Train model using reverse-translated Swiss-Prot sequences
        clf = train_kmer_classifier()
        
        # Test predictions
        test_kmers = generate_random_kmers(100)
        predictions = []
        for kmer in test_kmers:
            features = create_feature_vector(kmer)
            pred = clf.predict_proba([features])[0][1]
            predictions.append(pred)
        
        # Analyze results
        biological_kmers = [k for k, p in zip(test_kmers, predictions) if p > 0.8]
        logger.info(f"Found {len(biological_kmers)} likely biological k-mers")
        
    except Exception as e:
        logger.error("Analysis failed", exc_info=True)
        raise

# ...existing imports...

def generate_negative_kmers(k, n_samples=1000):
    """Generate artificial negative k-mer examples for supervised learning.
    
    Args:
        k (int): K-mer length
        n_samples (int): Number of negative examples to generate
        
    Returns:
        list: Negative k-mer examples that violate biological patterns
    """
    negative_kmers = []
    strategies = [
        'extreme_gc',
        'homopolymer',
        'invalid_motifs',
        'palindrome',
        'repeat_violate'
    ]
    
    samples_per_strategy = n_samples // len(strategies)
    
    for strategy in strategies:
        if strategy == 'extreme_gc':
            # Generate sequences with extreme GC content
            for _ in range(samples_per_strategy):
                if np.random.random() < 0.5:
                    # Very high GC (>80%)
                    seq = ''.join(np.random.choice(['G', 'C'], k))
                else:
                    # Very low GC (<20%)
                    seq = ''.join(np.random.choice(['A', 'T'], k))
                negative_kmers.append(seq)
                
        elif strategy == 'homopolymer':
            # Generate long homopolymer runs
            for _ in range(samples_per_strategy):
                base = np.random.choice(['A', 'T', 'G', 'C'])
                run_length = k // 2  # At least half homopolymer
                seq = base * run_length
                # Fill rest randomly
                seq += ''.join(np.random.choice(['A', 'T', 'G', 'C'], k - run_length))
                negative_kmers.append(seq)
                
        elif strategy == 'invalid_motifs':
            # Generate sequences with invalid biological motifs
            invalid_motifs = [
                'ATATAT',  # Invalid TATA box
                'GCGCGC',  # Unstable repeat
                'TTTAAA',  # Weak binding
                'CCCCCC'   # Structurally unstable
            ]
            for _ in range(samples_per_strategy):
                motif = np.random.choice(invalid_motifs)
                pos = np.random.randint(0, k - len(motif))
                seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], pos))
                seq += motif
                seq += ''.join(np.random.choice(['A', 'T', 'G', 'C'], k - pos - len(motif)))
                negative_kmers.append(seq)
                
        elif strategy == 'palindrome':
            # Generate sequences with too many palindromes
            for _ in range(samples_per_strategy):
                half = k // 2
                first_half = ''.join(np.random.choice(['A', 'T', 'G', 'C'], half))
                # Create perfect palindrome
                complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
                second_half = ''.join(complement[base] for base in reversed(first_half))
                seq = first_half + second_half
                if len(seq) < k:  # Add random base if k is odd
                    seq += np.random.choice(['A', 'T', 'G', 'C'])
                negative_kmers.append(seq)
                
        elif strategy == 'repeat_violate':
            # Generate sequences with invalid repeat patterns
            for _ in range(samples_per_strategy):
                repeat_unit = ''.join(np.random.choice(['A', 'T', 'G', 'C'], 3))
                repeats = k // len(repeat_unit)
                seq = repeat_unit * repeats
                if len(seq) < k:
                    seq += repeat_unit[:k-len(seq)]
                negative_kmers.append(seq)
    
    # Add violation of codon structure
    while len(negative_kmers) < n_samples:
        seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], k))
        # Ensure sequence disrupts codon periodicity
        if len(seq) >= 6:  # Need at least 2 codons
            # Insert base to shift reading frame
            pos = np.random.randint(3, len(seq)-3)
            seq = seq[:pos] + np.random.choice(['A', 'T', 'G', 'C']) + seq[pos:-1]
        negative_kmers.append(seq[:k])
    
    # Verify sequences are unique
    negative_kmers = list(set(negative_kmers))
    
    # If we lost some sequences due to duplicates, generate more
    while len(negative_kmers) < n_samples:
        seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], k))
        negative_kmers.append(seq)
    
    return negative_kmers[:n_samples]

def verify_negative_kmer(kmer):
    """Verify that a k-mer is likely to be a true negative example.
    
    Args:
        kmer (str): K-mer sequence to verify
        
    Returns:
        bool: True if sequence is likely a true negative
    """
    # Check for extreme GC content
    gc_content = (kmer.count('G') + kmer.count('C')) / len(kmer)
    if gc_content > 0.8 or gc_content < 0.2:
        return True
        
    # Check for homopolymer runs
    for base in 'ATGC':
        if base * (len(kmer)//2) in kmer:
            return True
    
    # Check for perfect palindromes of significant length
    half_len = len(kmer) // 2
    first_half = kmer[:half_len]
    second_half = kmer[-half_len:]
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    if all(complement.get(a) == b for a, b in zip(first_half, reversed(second_half))):
        return True
    
    # Check for excessive repeats
    for length in range(2, 6):
        for i in range(len(kmer)-length):
            pattern = kmer[i:i+length]
            if kmer.count(pattern) > 2:  # More than 2 occurrences of any pattern
                return True
    
    # Check for invalid biological motifs
    invalid_motifs = [
        'ATATAT', 'GCGCGC', 'TTTAAA', 'CCCCCC',
        'GGGGGG', 'AAAAAA', 'TTTTTT'
    ]
    if any(motif in kmer for motif in invalid_motifs):
        return True
    
    return False

# Update create_kmer_dataset to use new negative generation
def create_kmer_dataset(fasta_file="uniprot_sprot.fasta", k_size=31, n_samples=1000):
    """Create balanced dataset with artificial negative examples."""
    X, y = [], []
    
    # Generate positive examples from biological sequences
    biological_kmers = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        for i in range(len(seq) - k_size + 1):
            kmer = seq[i:i+k_size]
            if 'N' not in kmer and kmer not in biological_kmers:
                biological_kmers.add(kmer)
                features = create_feature_vector(kmer)
                X.append(features)
                y.append(1)
                
                if len(biological_kmers) >= n_samples:
                    break
        if len(biological_kmers) >= n_samples:
            break
    
    # Generate negative examples
    negative_kmers = generate_negative_kmers(k_size, n_samples)
    
    # Create features for negative examples
    for kmer in negative_kmers:
        if kmer not in biological_kmers:  # Avoid any overlap
            features = create_feature_vector(kmer)
            X.append(features)
            y.append(0)
    
    return np.array(X), np.array(y)

# Example usage in main
if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting k-mer analysis with artificial negative examples")
    
    try:
        # Generate balanced dataset
        X, y = create_kmer_dataset(n_samples=10000)
        
        # Train classifier
        clf = train_kmer_classifier()
        
        # Evaluate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)
        
        # Test predictions
        y_pred = clf.predict(X_test)
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
    except Exception as e:
        logger.error("Analysis failed", exc_info=True)
        raise

# ...existing code...

class DatasetValidator:
    """Validate and prepare datasets for k-mer classification."""
    
    def __init__(self, config=None):
        self.config = config or config_manager
        self.blast_db = None
        self.kmers_seen = BloomFilter(1000000, 5)
    
    def setup_blast_db(self, sequences):
        """Create temporary BLAST database for overlap checking."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:
            for i, seq in enumerate(sequences):
                temp_file.write(f">seq_{i}\n{seq}\n")
            temp_file.flush()
            
            try:
                subprocess.run(['makeblastdb', '-in', temp_file.name,
                              '-dbtype', 'nucl', '-out', 'temp_blast_db'],
                             check=True, capture_output=True)
                self.blast_db = 'temp_blast_db'
            except subprocess.CalledProcessError as e:
                logger.warning(f"BLAST database creation failed: {e}")
                self.blast_db = None

    def check_sequence_overlap(self, sequence):
        """Check if sequence has significant overlap with reference database."""
        if not self.blast_db:
            return False
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as query_file:
            query_file.write(f">query\n{sequence}\n")
            query_file.flush()
            
            try:
                result = subprocess.run(['blastn', '-query', query_file.name,
                                      '-db', self.blast_db, '-outfmt', '6',
                                      '-perc_identity', '80'],
                                     check=True, capture_output=True, text=True)
                return bool(result.stdout.strip())
            except subprocess.CalledProcessError:
                return False

    def prepare_dataset(self, fasta_file, k_size=31, negative_sample_size=None):
        """Prepare validated dataset with both real and synthetic sequences."""
        biological_kmers = []
        synthetic_kmers = []
        
        # Load biological sequences
        logger.info("Loading biological sequences...")
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq)
            for i in range(len(seq) - k_size + 1):
                kmer = seq[i:i+k_size]
                if 'N' not in kmer and kmer not in self.kmers_seen:
                    biological_kmers.append(kmer)
                    self.kmers_seen.add(kmer)
        
        # Setup BLAST database for overlap checking
        self.setup_blast_db(biological_kmers)
        
        # Generate synthetic sequences
        logger.info("Generating synthetic sequences...")
        negative_count = 0
        while negative_count < len(biological_kmers):
            kmer = self._generate_synthetic_kmer(k_size)
            if (kmer not in self.kmers_seen and 
                verify_negative_kmer(kmer) and 
                not self.check_sequence_overlap(kmer)):
                synthetic_kmers.append(kmer)
                self.kmers_seen.add(kmer)
                negative_count += 1
        
        return biological_kmers, synthetic_kmers
    
    def _generate_synthetic_kmer(self, k_size):
        """Generate a single synthetic k-mer using various strategies."""
        strategies = [
            lambda: self._extreme_gc_kmer(k_size),
            lambda: self._homopolymer_kmer(k_size),
            lambda: self._invalid_motif_kmer(k_size),
            lambda: self._repeat_violation_kmer(k_size)
        ]
        return np.random.choice(strategies)()

    # ... (keep existing synthetic sequence generation methods) ...

def create_balanced_dataset(fasta_file, k_size=31, test_size=0.2):
    """Create balanced dataset with validation."""
    validator = DatasetValidator()
    biological_kmers, synthetic_kmers = validator.prepare_dataset(fasta_file, k_size)
    
    # Create features
    X_bio = np.array([create_feature_vector(kmer) for kmer in biological_kmers])
    X_syn = np.array([create_feature_vector(kmer) for kmer in synthetic_kmers])
    
    # Create labels
    y_bio = np.ones(len(X_bio))
    y_syn = np.zeros(len(X_syn))
    
    # Combine and shuffle
    X = np.vstack([X_bio, X_syn])
    y = np.hstack([y_bio, y_syn])
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y)
    
    return X_train, X_test, y_train, y_test

class ModelEvaluator:
    """Comprehensive model evaluation tools."""
    
    @staticmethod
    def evaluate_model(clf, X_test, y_test):
        """Perform comprehensive model evaluation."""
        # Get predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob),
            'average_precision': average_precision_score(y_test, y_prob)
        }
        
        # Generate curves
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        return metrics, (fpr, tpr), (precision, recall)
    
    @staticmethod
    def plot_learning_curve(clf, X, y, cv=5):
        """Plot learning curve to assess model behavior with dataset size."""
        train_sizes, train_scores, test_scores = learning_curve(
            clf, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

def train_kmer_classifier(fasta_file=None, k_size=31):
    """Enhanced training workflow with validation."""
    if fasta_file is None:
        fasta_file = config_manager.get('PATHS', 'databases', 'biological_sequences')
    
    # Prepare dataset
    X_train, X_test, y_train, y_test = create_balanced_dataset(fasta_file, k_size)
    
    # Initialize classifier with early stopping
    clf = SGDClassifier(
        loss='log_loss',
        learning_rate='adaptive',
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42
    )
    
    # Train model
    clf.fit(X_train, y_train)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    metrics, roc_curve, pr_curve = evaluator.evaluate_model(clf, X_test, y_test)
    
    # Log results
    logger.info("\nModel Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Plot learning curve
    evaluator.plot_learning_curve(clf, X_train, y_train)
    
    return clf, metrics

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting enhanced k-mer analysis workflow")
    
    try:
        # Train and evaluate model
        clf, metrics = train_kmer_classifier()
        
        # Save model if performance is good
        if metrics['f1'] > 0.8:
            model_handler = ModelPersistence()
            model_handler.save_model(clf, metadata={'metrics': metrics})
            logger.info("Model saved successfully")
            
    except Exception as e:
        logger.error("Analysis failed", exc_info=True)
        raise
