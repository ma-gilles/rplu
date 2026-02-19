#!/usr/bin/env python3
"""
Test comparing CURIncremental vs PivotedQRCUR vs randomized_svd
on Toeplitz 3D problems with plots.
"""

import os
os.environ["JAX_CAPTURED_CONSTANTS_REPORT_FRAMES"] = "-1"  # Report where constants are captured

import json

import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

from kernels import drifted_aniso_gaussian_kernel3d
from conv_nd_operator import ConvNDOperator
from cur import CURIncremental
from randomized_svd import randomized_svd
from matrix_classes import SparseMatrixOperator, AOperator
from pivoted_qr_cur_multi_rank import compute_pivoted_qr_cur_multi_rank
from pivoted_qr import PivotedQR
from frobenius_error import compute_frobenius_error_cur, compute_frobenius_error_svd
from norm_estimation import lanczos_2norm
from datetime import datetime

def save_comparison_results(results, metadata, output_file, verbose=True):
    """
    Save comparison results to a JSON file for later analysis.
    
    Parameters
    ----------
    results : dict
        Dictionary with method names as keys and result dictionaries as values
    metadata : dict
        Metadata about the experiment (dim, problem_type, etc.)
    output_file : str
        Path to output JSON file
    verbose : bool
        If True, print save message
    """
    # Convert results to JSON-serializable format
    def convert_to_serializable(obj):
        if hasattr(obj, '__class__') and 'jax' in str(type(obj)).lower():
            try:
                obj = np.asarray(obj)
            except:
                return str(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            if obj.size == 1:
                return float(obj.item()) if obj.dtype.kind == 'f' else int(obj.item())
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            try:
                arr = np.asarray(obj)
                if arr.size == 1:
                    return float(arr.item()) if arr.dtype.kind == 'f' else int(arr.item())
                return arr.tolist()
            except:
                return str(obj)
    
    serializable_results = convert_to_serializable(results)
    serializable_metadata = convert_to_serializable(metadata)
    
    output_data = {
        'metadata': serializable_metadata,
        'timestamp': datetime.now().isoformat(),
        'results': serializable_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    if verbose:
        print(f"Results saved to: {output_file}")
    
    return output_file


import tarfile
import requests
from scipy.io import mmread
import scipy.sparse as sp
from scipy.linalg import qr


def create_dense_matrix_with_slow_decay(n=1000, random_state=42, final_ratio=1e-8):
    """
    Create a dense n x n matrix with slowly decaying singular values.
    The singular values decay from 1 to approximately final_ratio.
    
    Parameters
    ----------
    n : int
        Matrix size (n x n)
    random_state : int
        Random seed for generating random orthogonal matrices
    final_ratio : float
        Ratio of last to first singular value (default: 1e-8 for slower decay)
        
    Returns
    -------
    A : jnp.ndarray
        Dense matrix with shape (n, n) and slowly decaying singular values
    """
    rng = np.random.RandomState(random_state)
    
    # Create singular values that decay exponentially from 1 to ~final_ratio
    # Use exponential decay: s[i] = exp(-alpha * i) where alpha is chosen so that
    # s[n-1] / s[0] ≈ final_ratio
    # exp(-alpha * (n-1)) = final_ratio
    # -alpha * (n-1) = log(final_ratio)
    # alpha = -log(final_ratio) / (n-1)
    alpha = -np.log(final_ratio) / (n - 1)
    singular_values = np.exp(-alpha * np.arange(n))
    # Normalize so first singular value is 1 (already is, but ensure it)
    singular_values = singular_values / singular_values[0]
    
    # Generate random orthogonal matrices U and V
    # Use QR decomposition of random matrices to get orthogonal matrices
    U, _ = np.linalg.qr(rng.randn(n, n))
    V, _ = np.linalg.qr(rng.randn(n, n))
    
    # Construct matrix A = U @ diag(s) @ V^T
    A = U @ np.diag(singular_values) @ V.T
    
    print(f"Created dense {n}x{n} matrix")
    print(f"  Singular value range: {singular_values[0]:.2e} to {singular_values[-1]:.2e}")
    print(f"  Ratio (last/first): {singular_values[-1]/singular_values[0]:.2e}")
    
    return jnp.asarray(A, dtype=jnp.float64)


def create_line_graph_adjacency(n):
    """
    Create adjacency matrix for a directed path graph (line graph) of n nodes.
    
    Parameters
    ----------
    n : int
        Number of nodes in the path graph
    
    Returns
    -------
    A_scipy : scipy.sparse.csr_matrix
        Adjacency matrix of the directed line graph (n x n)
        Edges go from node i to node i+1 (only one direction)
    """
    # Directed path graph: nodes connected in a line 1->2->3->...->n
    # Adjacency matrix has 1s only on superdiagonal (edges from i to i+1)
    A = sp.diags([1], [1], shape=(n, n), format='csr')
    return A


def download_and_extract_graph(url: str, tar_path: str, extract_dir: str):
    """Download and extract a graph matrix from SuiteSparse."""
    if not os.path.exists(tar_path):
        print(f"Downloading {url} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    
    if not os.path.exists(extract_dir):
        print(f"Extracting {tar_path} ...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall()
    else:
        print(f"Directory {extract_dir} already exists, skipping extract.")


def load_graph_matrix(graph_name='mawi', graphs_dir=None):
    """
    Load a graph matrix from SuiteSparse Matrix Collection.
    
    Parameters
    ----------
    graph_name : str
        Name of graph: 'mawi', 'snap', 'web-google', or one of the 10 large graphs
    graphs_dir : str, optional
        Directory where graphs are stored. If None, uses current directory.
        If provided, looks for graphs in this directory first.
    
    Returns
    -------
    A_op : SparseMatrixOperator
        The graph matrix as an AOperator
    graph_info : dict
        Information about the graph
    """
    # Map of graph names to their metadata
    graph_urls = {
        'mawi': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512012345.tar.gz",
            'tar_path': "mawi_201512012345.tar.gz",
            'extract_dir': "mawi_201512012345",
            'mtx_path': "mawi_201512012345/mawi_201512012345.mtx"
        },
        'snap': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-GrQc.tar.gz",
            'tar_path': "ca-GrQc.tar.gz",
            'extract_dir': "ca-GrQc",
            'mtx_path': "ca-GrQc/ca-GrQc.mtx"
        },
        'web-google': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Web/web-Google.tar.gz",
            'tar_path': "web-Google.tar.gz",
            'extract_dir': "web-Google",
            'mtx_path': "web-Google/web-Google.mtx"
        },
        'roadnet-pa': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-PA.tar.gz",
            'tar_path': "roadNet-PA.tar.gz",
            'extract_dir': "roadNet-PA",
            'mtx_path': "roadNet-PA/roadNet-PA.mtx"
        },
        'roadnet-ca': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz",
            'tar_path': "roadNet-CA.tar.gz",
            'extract_dir': "roadNet-CA",
            'mtx_path': "roadNet-CA/roadNet-CA.mtx"
        },
        'soc-livejournal': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz",
            'tar_path': "soc-LiveJournal1.tar.gz",
            'extract_dir': "soc-LiveJournal1",
            'mtx_path': "soc-LiveJournal1/soc-LiveJournal1.mtx"
        },
        'wiki-talk': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Talk.tar.gz",
            'tar_path': "wiki-Talk.tar.gz",
            'extract_dir': "wiki-Talk",
            'mtx_path': "wiki-Talk/wiki-Talk.mtx"
        },
        # New large graphs from download_graphs.py
        'mawi_201512020330': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020330.tar.gz",
            'tar_path': "mawi_201512020330.tar.gz",
            'extract_dir': "mawi_201512020330",
            'mtx_path': "mawi_201512020330/mawi_201512020330.mtx"
        },
        'kmer_V1r': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/GenBank/kmer_V1r.tar.gz",
            'tar_path': "kmer_V1r.tar.gz",
            'extract_dir': "kmer_V1r",
            'mtx_path': "kmer_V1r/kmer_V1r.mtx"
        },
        'kmer_A2a': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/GenBank/kmer_A2a.tar.gz",
            'tar_path': "kmer_A2a.tar.gz",
            'extract_dir': "kmer_A2a",
            'mtx_path': "kmer_A2a/kmer_A2a.mtx"
        },
        'kmer_P1a': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/GenBank/kmer_P1a.tar.gz",
            'tar_path': "kmer_P1a.tar.gz",
            'extract_dir': "kmer_P1a",
            'mtx_path': "kmer_P1a/kmer_P1a.mtx"
        },
        'mawi_201512020130': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020130.tar.gz",
            'tar_path': "mawi_201512020130.tar.gz",
            'extract_dir': "mawi_201512020130",
            'mtx_path': "mawi_201512020130/mawi_201512020130.mtx"
        },
        'webbase-2001': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/webbase-2001.tar.gz",
            'tar_path': "webbase-2001.tar.gz",
            'extract_dir': "webbase-2001",
            'mtx_path': "webbase-2001/webbase-2001.mtx"
        },
        'Spielman_k600': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/FlowIPM22/Spielman_k600.tar.gz",
            'tar_path': "Spielman_k600.tar.gz",
            'extract_dir': "Spielman_k600",
            'mtx_path': "Spielman_k600/Spielman_k600.mtx"
        },
        'mawi_201512020030': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020030.tar.gz",
            'tar_path': "mawi_201512020030.tar.gz",
            'extract_dir': "mawi_201512020030",
            'mtx_path': "mawi_201512020030/mawi_201512020030.mtx"
        },
        'kmer_U1a': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/GenBank/kmer_U1a.tar.gz",
            'tar_path': "kmer_U1a.tar.gz",
            'extract_dir': "kmer_U1a",
            'mtx_path': "kmer_U1a/kmer_U1a.mtx"
        },
        'kmer_V2a': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/GenBank/kmer_V2a.tar.gz",
            'tar_path': "kmer_V2a.tar.gz",
            'extract_dir': "kmer_V2a",
            'mtx_path': "kmer_V2a/kmer_V2a.mtx"
        },
        'europe_osm': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/europe_osm.tar.gz",
            'tar_path': "europe_osm.tar.gz",
            'extract_dir': "europe_osm",
            'mtx_path': "europe_osm/europe_osm.mtx"
        },
        'sk-2005': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/sk-2005.tar.gz",
            'tar_path': "sk-2005.tar.gz",
            'extract_dir': "sk-2005",
            'mtx_path': "sk-2005/sk-2005.mtx"
        },
        'twitter7': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/twitter7.tar.gz",
            'tar_path': "twitter7.tar.gz",
            'extract_dir': "twitter7",
            'mtx_path': "twitter7/twitter7.mtx"
        },
        # Large web graphs (LAW) - typically have good low-rank structure due to link patterns
        'uk-2005': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/uk-2005.tar.gz",
            'tar_path': "uk-2005.tar.gz",
            'extract_dir': "uk-2005",
            'mtx_path': "uk-2005/uk-2005.mtx"
        },
        'it-2004': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/it-2004.tar.gz",
            'tar_path': "it-2004.tar.gz",
            'extract_dir': "it-2004",
            'mtx_path': "it-2004/it-2004.mtx"
        },
        'indochina-2004': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/indochina-2004.tar.gz",
            'tar_path': "indochina-2004.tar.gz",
            'extract_dir': "indochina-2004",
            'mtx_path': "indochina-2004/indochina-2004.mtx"
        },
        'arabic-2005': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/arabic-2005.tar.gz",
            'tar_path': "arabic-2005.tar.gz",
            'extract_dir': "arabic-2005",
            'mtx_path': "arabic-2005/arabic-2005.mtx"
        },
        # Social networks (SNAP) - community structure = good low-rank
        'com-Orkut': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Orkut.tar.gz",
            'tar_path': "com-Orkut.tar.gz",
            'extract_dir': "com-Orkut",
            'mtx_path': "com-Orkut/com-Orkut.mtx"
        },
        'com-LiveJournal': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-LiveJournal.tar.gz",
            'tar_path': "com-LiveJournal.tar.gz",
            'extract_dir': "com-LiveJournal",
            'mtx_path': "com-LiveJournal/com-LiveJournal.mtx"
        },
        'com-Friendster': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Friendster.tar.gz",
            'tar_path': "com-Friendster.tar.gz",
            'extract_dir': "com-Friendster",
            'mtx_path': "com-Friendster/com-Friendster.mtx"
        },
        # Collaboration network
        'hollywood-2009': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/hollywood-2009.tar.gz",
            'tar_path': "hollywood-2009.tar.gz",
            'extract_dir': "hollywood-2009",
            'mtx_path': "hollywood-2009/hollywood-2009.mtx"
        },
        # Additional large graphs
        'road_usa': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/road_usa.tar.gz",
            'tar_path': "road_usa.tar.gz",
            'extract_dir': "road_usa",
            'mtx_path': "road_usa/road_usa.mtx"
        },
        'wb-edu': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Gleich/wb-edu.tar.gz",
            'tar_path': "wb-edu.tar.gz",
            'extract_dir': "wb-edu",
            'mtx_path': "wb-edu/wb-edu.mtx"
        },
        'uk-2002': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/uk-2002.tar.gz",
            'tar_path': "uk-2002.tar.gz",
            'extract_dir': "uk-2002",
            'mtx_path': "uk-2002/uk-2002.mtx"
        },
        'eu-2005': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/eu-2005.tar.gz",
            'tar_path': "eu-2005.tar.gz",
            'extract_dir': "eu-2005",
            'mtx_path': "eu-2005/eu-2005.mtx"
        },
        'com-Friendster': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Friendster.tar.gz",
            'tar_path': "com-Friendster.tar.gz",
            'extract_dir': "com-Friendster",
            'mtx_path': "com-Friendster/com-Friendster.mtx"
        },
        'uk-2007-05': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/uk-2007-05.tar.gz",
            'tar_path': "uk-2007-05.tar.gz",
            'extract_dir': "uk-2007-05",
            'mtx_path': "uk-2007-05/uk-2007-05.mtx"
        },
        # Wikipedia link graph - known to have good spectral structure
        'wikipedia-20070206': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Gleich/wikipedia-20070206.tar.gz",
            'tar_path': "wikipedia-20070206.tar.gz",
            'extract_dir': "wikipedia-20070206",
            'mtx_path': "wikipedia-20070206/wikipedia-20070206.mtx"
        },
        # Citation networks - typically have good rank decay
        'cit-Patents': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-Patents.tar.gz",
            'tar_path': "cit-Patents.tar.gz",
            'extract_dir': "cit-Patents",
            'mtx_path': "cit-Patents/cit-Patents.mtx"
        },
        # Social networks
        'soc-pokec': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-pokec.tar.gz",
            'tar_path': "soc-pokec.tar.gz",
            'extract_dir': "soc-pokec",
            'mtx_path': "soc-pokec/soc-pokec.mtx"
        },
        # Amazon co-purchasing network
        'amazon0601': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0601.tar.gz",
            'tar_path': "amazon0601.tar.gz",
            'extract_dir': "amazon0601",
            'mtx_path': "amazon0601/amazon0601.mtx"
        },
        # =================================================================
        # Small graphs from EG22 paper (Table 2) - good for quick testing
        # =================================================================
        # Chemical simulation (Grund group)
        'bayer01': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Grund/bayer01.tar.gz",
            'tar_path': "bayer01.tar.gz",
            'extract_dir': "bayer01",
            'mtx_path': "bayer01/bayer01.mtx"
        },
        # Structural (Boeing group)
        'bcsstk36': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/bcsstk36.tar.gz",
            'tar_path': "bcsstk36.tar.gz",
            'extract_dir': "bcsstk36",
            'mtx_path': "bcsstk36/bcsstk36.mtx"
        },
        # Nonlinear optimization (Schenk_IBMNA group)
        'c-67b': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Schenk_IBMNA/c-67b.tar.gz",
            'tar_path': "c-67b.tar.gz",
            'extract_dir': "c-67b",
            'mtx_path': "c-67b/c-67b.mtx"
        },
        # Nonlinear optimization (GHS_indef group)
        'c-69': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/c-69.tar.gz",
            'tar_path': "c-69.tar.gz",
            'extract_dir': "c-69",
            'mtx_path': "c-69/c-69.mtx"
        },
        # Structural (TKK group)
        'cbuckle': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/TKK/cbuckle.tar.gz",
            'tar_path': "cbuckle.tar.gz",
            'extract_dir': "cbuckle",
            'mtx_path': "cbuckle/cbuckle.mtx"
        },
        # Structural (GHS_psdef group)
        'crankseg_2': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/crankseg_2.tar.gz",
            'tar_path': "crankseg_2.tar.gz",
            'extract_dir': "crankseg_2",
            'mtx_path': "crankseg_2/crankseg_2.mtx"
        },
        # Structural (Boeing group)
        'ct20stif': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/ct20stif.tar.gz",
            'tar_path': "ct20stif.tar.gz",
            'extract_dir': "ct20stif",
            'mtx_path': "ct20stif/ct20stif.mtx"
        },
        # Economics (Hollinger group)
        'g7jac200sc': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Hollinger/g7jac200sc.tar.gz",
            'tar_path': "g7jac200sc.tar.gz",
            'extract_dir': "g7jac200sc",
            'mtx_path': "g7jac200sc/g7jac200sc.mtx"
        },
        # Fluid dynamics (Simon group)
        'venkat01': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Simon/venkat01.tar.gz",
            'tar_path': "venkat01.tar.gz",
            'extract_dir': "venkat01",
            'mtx_path': "venkat01/venkat01.mtx"
        },
        # =================================================================
        # Large graphs (>1M nodes) - commonly used in literature
        # =================================================================
        # Social Networks with Community Structure
        'com-Orkut': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Orkut.tar.gz",
            'tar_path': "com-Orkut.tar.gz",
            'extract_dir': "com-Orkut",
            'mtx_path': "com-Orkut/com-Orkut.mtx"
        },
        'soc-LiveJournal1': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz",
            'tar_path': "soc-LiveJournal1.tar.gz",
            'extract_dir': "soc-LiveJournal1",
            'mtx_path': "soc-LiveJournal1/soc-LiveJournal1.mtx"
        },
        'as-Skitter': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/as-Skitter.tar.gz",
            'tar_path': "as-Skitter.tar.gz",
            'extract_dir': "as-Skitter",
            'mtx_path': "as-Skitter/as-Skitter.mtx"
        },
        # LAW Web Graphs (hierarchical structure)
        'ljournal-2008': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/ljournal-2008.tar.gz",
            'tar_path': "ljournal-2008.tar.gz",
            'extract_dir': "ljournal-2008",
            'mtx_path': "ljournal-2008/ljournal-2008.mtx"
        },
        'in-2004': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/LAW/in-2004.tar.gz",
            'tar_path': "in-2004.tar.gz",
            'extract_dir': "in-2004",
            'mtx_path': "in-2004/in-2004.mtx"
        },
        # DIMACS10 Benchmark Graphs
        'delaunay_n21': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n21.tar.gz",
            'tar_path': "delaunay_n21.tar.gz",
            'extract_dir': "delaunay_n21",
            'mtx_path': "delaunay_n21/delaunay_n21.mtx"
        },
        'delaunay_n22': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n22.tar.gz",
            'tar_path': "delaunay_n22.tar.gz",
            'extract_dir': "delaunay_n22",
            'mtx_path': "delaunay_n22/delaunay_n22.mtx"
        },
        'delaunay_n23': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n23.tar.gz",
            'tar_path': "delaunay_n23.tar.gz",
            'extract_dir': "delaunay_n23",
            'mtx_path': "delaunay_n23/delaunay_n23.mtx"
        },
        'hugebubbles-00000': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/hugebubbles-00000.tar.gz",
            'tar_path': "hugebubbles-00000.tar.gz",
            'extract_dir': "hugebubbles-00000",
            'mtx_path': "hugebubbles-00000/hugebubbles-00000.mtx"
        },
        'road_central': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/road_central.tar.gz",
            'tar_path': "road_central.tar.gz",
            'extract_dir': "road_central",
            'mtx_path': "road_central/road_central.mtx"
        },
        # Circuit Networks (hierarchical design)
        'circuit5M': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M.tar.gz",
            'tar_path': "circuit5M.tar.gz",
            'extract_dir': "circuit5M",
            'mtx_path': "circuit5M/circuit5M.mtx"
        },
        'FullChip': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/FullChip.tar.gz",
            'tar_path': "FullChip.tar.gz",
            'extract_dir': "FullChip",
            'mtx_path': "FullChip/FullChip.mtx"
        },
        'Freescale1': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/Freescale1.tar.gz",
            'tar_path': "Freescale1.tar.gz",
            'extract_dir': "Freescale1",
            'mtx_path': "Freescale1/Freescale1.mtx"
        },
        'memchip': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/memchip.tar.gz",
            'tar_path': "memchip.tar.gz",
            'extract_dir': "memchip",
            'mtx_path': "memchip/memchip.mtx"
        },
        # Thermal/FEM (spectral structure)
        'thermal2': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal2.tar.gz",
            'tar_path': "thermal2.tar.gz",
            'extract_dir': "thermal2",
            'mtx_path': "thermal2/thermal2.mtx"
        },
        'G3_circuit': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/AMD/G3_circuit.tar.gz",
            'tar_path': "G3_circuit.tar.gz",
            'extract_dir': "G3_circuit",
            'mtx_path': "G3_circuit/G3_circuit.mtx"
        },
        # Wikipedia graphs (topic structure)
        'wikipedia-20051105': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Gleich/wikipedia-20051105.tar.gz",
            'tar_path': "wikipedia-20051105.tar.gz",
            'extract_dir': "wikipedia-20051105",
            'mtx_path': "wikipedia-20051105/wikipedia-20051105.mtx"
        },
        'wikipedia-20060925': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Gleich/wikipedia-20060925.tar.gz",
            'tar_path': "wikipedia-20060925.tar.gz",
            'extract_dir': "wikipedia-20060925",
            'mtx_path': "wikipedia-20060925/wikipedia-20060925.mtx"
        },
        'wikipedia-20061104': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Gleich/wikipedia-20061104.tar.gz",
            'tar_path': "wikipedia-20061104.tar.gz",
            'extract_dir': "wikipedia-20061104",
            'mtx_path': "wikipedia-20061104/wikipedia-20061104.mtx"
        },
        # =================================================================
        # Additional interesting graphs - diverse types
        # =================================================================
        # DNA electrophoresis (very interesting structure)
        'cage15': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage15.tar.gz",
            'tar_path': "cage15.tar.gz",
            'extract_dir': "cage15",
            'mtx_path': "cage15/cage15.mtx"
        },
        'cage14': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage14.tar.gz",
            'tar_path': "cage14.tar.gz",
            'extract_dir': "cage14",
            'mtx_path': "cage14/cage14.mtx"
        },
        # Structural FEM (good spectral decay)
        'Flan_1565': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Flan_1565.tar.gz",
            'tar_path': "Flan_1565.tar.gz",
            'extract_dir': "Flan_1565",
            'mtx_path': "Flan_1565/Flan_1565.mtx"
        },
        'Serena': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Serena.tar.gz",
            'tar_path': "Serena.tar.gz",
            'extract_dir': "Serena",
            'mtx_path': "Serena/Serena.mtx"
        },
        'Geo_1438': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Geo_1438.tar.gz",
            'tar_path': "Geo_1438.tar.gz",
            'extract_dir': "Geo_1438",
            'mtx_path': "Geo_1438/Geo_1438.mtx"
        },
        'Hook_1498': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Hook_1498.tar.gz",
            'tar_path': "Hook_1498.tar.gz",
            'extract_dir': "Hook_1498",
            'mtx_path': "Hook_1498/Hook_1498.mtx"
        },
        'audikw_1': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/audikw_1.tar.gz",
            'tar_path': "audikw_1.tar.gz",
            'extract_dir': "audikw_1",
            'mtx_path': "audikw_1/audikw_1.mtx"
        },
        'ldoor': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz",
            'tar_path': "ldoor.tar.gz",
            'extract_dir': "ldoor",
            'mtx_path': "ldoor/ldoor.mtx"
        },
        'boneS10': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/boneS10.tar.gz",
            'tar_path': "boneS10.tar.gz",
            'extract_dir': "boneS10",
            'mtx_path': "boneS10/boneS10.mtx"
        },
        # Automotive crash simulation
        'af_shell10': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell10.tar.gz",
            'tar_path': "af_shell10.tar.gz",
            'extract_dir': "af_shell10",
            'mtx_path': "af_shell10/af_shell10.mtx"
        },
        # Citation/Co-authorship networks (bipartite-like, good low rank)
        'coPapersCiteseer': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coPapersCiteseer.tar.gz",
            'tar_path': "coPapersCiteseer.tar.gz",
            'extract_dir': "coPapersCiteseer",
            'mtx_path': "coPapersCiteseer/coPapersCiteseer.mtx"
        },
        'coPapersDBLP': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coPapersDBLP.tar.gz",
            'tar_path': "coPapersDBLP.tar.gz",
            'extract_dir': "coPapersDBLP",
            'mtx_path': "coPapersDBLP/coPapersDBLP.mtx"
        },
        # Delaunay triangulation (largest)
        'delaunay_n24': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n24.tar.gz",
            'tar_path': "delaunay_n24.tar.gz",
            'extract_dir': "delaunay_n24",
            'mtx_path': "delaunay_n24/delaunay_n24.mtx"
        },
        # Social networks
        'flickr': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Gleich/flickr.tar.gz",
            'tar_path': "flickr.tar.gz",
            'extract_dir': "flickr",
            'mtx_path': "flickr/flickr.mtx"
        },
        'higgs-twitter': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/higgs-twitter.tar.gz",
            'tar_path': "higgs-twitter.tar.gz",
            'extract_dir': "higgs-twitter",
            'mtx_path': "higgs-twitter/higgs-twitter.mtx"
        },
        # Web graphs
        'web-BerkStan': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-BerkStan.tar.gz",
            'tar_path': "web-BerkStan.tar.gz",
            'extract_dir': "web-BerkStan",
            'mtx_path': "web-BerkStan/web-BerkStan.mtx"
        },
        'web-Google': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Google.tar.gz",
            'tar_path': "web-Google.tar.gz",
            'extract_dir': "web-Google",
            'mtx_path': "web-Google/web-Google.mtx"
        },
        # Road network
        'roadNet-CA': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz",
            'tar_path': "roadNet-CA.tar.gz",
            'extract_dir': "roadNet-CA",
            'mtx_path': "roadNet-CA/roadNet-CA.mtx"
        },
        # =================================================================
        # Gigantic diverse graphs (10M+ rows)
        # =================================================================
        # Optimization (interior point) - interesting structure
        'nlpkkt240': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt240.tar.gz",
            'tar_path': "nlpkkt240.tar.gz",
            'extract_dir': "nlpkkt240",
            'mtx_path': "nlpkkt240/nlpkkt240.mtx"
        },
        'nlpkkt200': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt200.tar.gz",
            'tar_path': "nlpkkt200.tar.gz",
            'extract_dir': "nlpkkt200",
            'mtx_path': "nlpkkt200/nlpkkt200.mtx"
        },
        'nlpkkt160': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt160.tar.gz",
            'tar_path': "nlpkkt160.tar.gz",
            'extract_dir': "nlpkkt160",
            'mtx_path': "nlpkkt160/nlpkkt160.mtx"
        },
        'nlpkkt120': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt120.tar.gz",
            'tar_path': "nlpkkt120.tar.gz",
            'extract_dir': "nlpkkt120",
            'mtx_path': "nlpkkt120/nlpkkt120.mtx"
        },
        # Structural FEM (Janna)
        'Cube_Coup_dt0': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Cube_Coup_dt0.tar.gz",
            'tar_path': "Cube_Coup_dt0.tar.gz",
            'extract_dir': "Cube_Coup_dt0",
            'mtx_path': "Cube_Coup_dt0/Cube_Coup_dt0.mtx"
        },
        'Long_Coup_dt0': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Long_Coup_dt0.tar.gz",
            'tar_path': "Long_Coup_dt0.tar.gz",
            'extract_dir': "Long_Coup_dt0",
            'mtx_path': "Long_Coup_dt0/Long_Coup_dt0.mtx"
        },
        'Emilia_923': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Emilia_923.tar.gz",
            'tar_path': "Emilia_923.tar.gz",
            'extract_dir': "Emilia_923",
            'mtx_path': "Emilia_923/Emilia_923.mtx"
        },
        'StocF-1465': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Janna/StocF-1465.tar.gz",
            'tar_path': "StocF-1465.tar.gz",
            'extract_dir': "StocF-1465",
            'mtx_path': "StocF-1465/StocF-1465.mtx"
        },
        # MAWI internet traffic (good low-rank per user)
        'mawi_201512020000': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020000.tar.gz",
            'tar_path': "mawi_201512020000.tar.gz",
            'extract_dir': "mawi_201512020000",
            'mtx_path': "mawi_201512020000/mawi_201512020000.mtx"
        },
        'mawi_201512012345': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512012345.tar.gz",
            'tar_path': "mawi_201512012345.tar.gz",
            'extract_dir': "mawi_201512012345",
            'mtx_path': "mawi_201512012345/mawi_201512012345.mtx"
        },
        # Bubble simulation
        'hugebubbles-00010': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/hugebubbles-00010.tar.gz",
            'tar_path': "hugebubbles-00010.tar.gz",
            'extract_dir': "hugebubbles-00010",
            'mtx_path': "hugebubbles-00010/hugebubbles-00010.mtx"
        },
        'hugebubbles-00020': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/hugebubbles-00020.tar.gz",
            'tar_path': "hugebubbles-00020.tar.gz",
            'extract_dir': "hugebubbles-00020",
            'mtx_path': "hugebubbles-00020/hugebubbles-00020.mtx"
        },
        # More circuit networks
        'circuit5M_dc': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M_dc.tar.gz",
            'tar_path': "circuit5M_dc.tar.gz",
            'extract_dir': "circuit5M_dc",
            'mtx_path': "circuit5M_dc/circuit5M_dc.mtx"
        },
        'Freescale2': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/Freescale2.tar.gz",
            'tar_path': "Freescale2.tar.gz",
            'extract_dir': "Freescale2",
            'mtx_path': "Freescale2/Freescale2.mtx"
        },
        'Hamrle3': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Hamrle/Hamrle3.tar.gz",
            'tar_path': "Hamrle3.tar.gz",
            'extract_dir': "Hamrle3",
            'mtx_path': "Hamrle3/Hamrle3.mtx"
        },
        'rajat31': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Rajat/rajat31.tar.gz",
            'tar_path': "rajat31.tar.gz",
            'extract_dir': "rajat31",
            'mtx_path': "rajat31/rajat31.mtx"
        },
        # Wikipedia talk network
        'wiki-Talk': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Talk.tar.gz",
            'tar_path': "wiki-Talk.tar.gz",
            'extract_dir': "wiki-Talk",
            'mtx_path': "wiki-Talk/wiki-Talk.mtx"
        },
        # Rectangular relational (bipartite)
        'relat9': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/JGD_Relat/relat9.tar.gz",
            'tar_path': "relat9.tar.gz",
            'extract_dir': "relat9",
            'mtx_path': "relat9/relat9.mtx"
        },
        # More Delaunay
        'delaunay_n20': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n20.tar.gz",
            'tar_path': "delaunay_n20.tar.gz",
            'extract_dir': "delaunay_n20",
            'mtx_path': "delaunay_n20/delaunay_n20.mtx"
        },
        # =================================================================
        # Diverse sparse matrices (not just graphs)
        # =================================================================
        # Citation network (Pajek)
        'patents': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Pajek/patents.tar.gz",
            'tar_path': "patents.tar.gz",
            'extract_dir': "patents",
            'mtx_path': "patents/patents.mtx"
        },
        # Linear programming (Mittelmann)
        'cont1_l': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Mittelmann/cont1_l.tar.gz",
            'tar_path': "cont1_l.tar.gz",
            'extract_dir': "cont1_l",
            'mtx_path': "cont1_l/cont1_l.mtx"
        },
        'cont11_l': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Mittelmann/cont11_l.tar.gz",
            'tar_path': "cont11_l.tar.gz",
            'extract_dir': "cont11_l",
            'mtx_path': "cont11_l/cont11_l.mtx"
        },
        'stormG2_1000': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Mittelmann/stormG2_1000.tar.gz",
            'tar_path': "stormG2_1000.tar.gz",
            'extract_dir': "stormG2_1000",
            'mtx_path': "stormG2_1000/stormG2_1000.mtx"
        },
        # Actor-movie bipartite (Pajek)
        'IMDB': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Pajek/IMDB.tar.gz",
            'tar_path': "IMDB.tar.gz",
            'extract_dir': "IMDB",
            'mtx_path': "IMDB/IMDB.mtx"
        },
        # Graph combinatorics (JGD_Margulies)
        'wheel_601': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/JGD_Margulies/wheel_601.tar.gz",
            'tar_path': "wheel_601.tar.gz",
            'extract_dir': "wheel_601",
            'mtx_path': "wheel_601/wheel_601.mtx"
        },
        # Relational (JGD_Relat) - rectangular
        'rel9': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/JGD_Relat/rel9.tar.gz",
            'tar_path': "rel9.tar.gz",
            'extract_dir': "rel9",
            'mtx_path': "rel9/rel9.mtx"
        },
        # Circuit simulation (ATandT)
        'pre2': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/ATandT/pre2.tar.gz",
            'tar_path': "pre2.tar.gz",
            'extract_dir': "pre2",
            'mtx_path': "pre2/pre2.mtx"
        },
        # Structural FEM (Boeing)
        'pwtk': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz",
            'tar_path': "pwtk.tar.gz",
            'extract_dir': "pwtk",
            'mtx_path': "pwtk/pwtk.mtx"
        },
        # Structural (Chen)
        'pkustk14': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Chen/pkustk14.tar.gz",
            'tar_path': "pkustk14.tar.gz",
            'extract_dir': "pkustk14",
            'mtx_path': "pkustk14/pkustk14.mtx"
        },
        # Structural (Kim)
        'kim2': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Kim/kim2.tar.gz",
            'tar_path': "kim2.tar.gz",
            'extract_dir': "kim2",
            'mtx_path': "kim2/kim2.mtx"
        },
        # DNA electrophoresis (vanHeukelum)
        'cage13': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage13.tar.gz",
            'tar_path': "cage13.tar.gz",
            'extract_dir': "cage13",
            'mtx_path': "cage13/cage13.mtx"
        },
        # Biomedical/Torso (Norris)
        'torso3': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Norris/torso3.tar.gz",
            'tar_path': "torso3.tar.gz",
            'extract_dir': "torso3",
            'mtx_path': "torso3/torso3.mtx"
        },
        # Automotive shell (Schenk_AFE)
        'af_shell9': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell9.tar.gz",
            'tar_path': "af_shell9.tar.gz",
            'extract_dir': "af_shell9",
            'mtx_path': "af_shell9/af_shell9.mtx"
        },
        # Mechanical (Rothberg)
        'gearbox': {
            'url': "https://suitesparse-collection-website.herokuapp.com/MM/Rothberg/gearbox.tar.gz",
            'tar_path': "gearbox.tar.gz",
            'extract_dir': "gearbox",
            'mtx_path': "gearbox/gearbox.mtx"
        },
    }
    
    if graph_name not in graph_urls:
        raise ValueError(f"Unknown graph name: {graph_name}. Choose from {list(graph_urls.keys())}")
    
    info = graph_urls[graph_name]
    
    # If graphs_dir is provided, look there first
    if graphs_dir and os.path.exists(graphs_dir):
        # Check if graph is already extracted in graphs_dir
        extract_dir_in_graphs = os.path.join(graphs_dir, info['extract_dir'])
        mtx_path_in_graphs = os.path.join(extract_dir_in_graphs, os.path.basename(info['mtx_path']))
        
        if os.path.exists(mtx_path_in_graphs):
            print(f"Using graph from {graphs_dir}")
            mtx_path = mtx_path_in_graphs
        else:
            # Check if tar file exists
            tar_path_in_graphs = os.path.join(graphs_dir, info['tar_path'])
            if os.path.exists(tar_path_in_graphs):
                print(f"Extracting graph from {tar_path_in_graphs}")
                with tarfile.open(tar_path_in_graphs, "r:gz") as tf:
                    tf.extractall(graphs_dir)
                mtx_path = os.path.join(extract_dir_in_graphs, os.path.basename(info['mtx_path']))
            else:
                # Download to graphs_dir
                print(f"Downloading graph to {graphs_dir}")
                download_and_extract_graph(info['url'], tar_path_in_graphs, extract_dir_in_graphs)
                mtx_path = os.path.join(extract_dir_in_graphs, os.path.basename(info['mtx_path']))
    else:
        # Original behavior: download and extract in current directory
        download_and_extract_graph(info['url'], info['tar_path'], info['extract_dir'])
        mtx_path = os.path.join(info['extract_dir'], os.path.basename(info['mtx_path']))
    
    print(f"Reading Matrix Market file {mtx_path} ...")
    A_scipy = mmread(mtx_path)
    A_scipy = A_scipy.tocsr()
    
    # Wrap as SparseMatrixOperator
    A_op = SparseMatrixOperator(A_scipy, dtype=jnp.float64)
    
    graph_info = {
        'name': graph_name,
        'shape': A_op.shape,
        'nnz': A_scipy.nnz,
        'sparsity': 1.0 - (A_scipy.nnz / (A_op.shape[0] * A_op.shape[1]))
    }
    
    print(f"Graph matrix: shape={A_op.shape}, nnz={A_scipy.nnz}, sparsity={graph_info['sparsity']:.4f}")
    
    return A_op, graph_info


def safe_clear_jax_caches():
    """Safely clear JAX caches, handling cases where it's not available or has issues."""
    try:
        jax.clear_caches()
    except AttributeError:
        # jax.clear_caches() doesn't exist in this JAX version
        pass
    except Exception as e:
        # Other errors (e.g., JAX internal errors) - just ignore
        pass


def estimate_approximation_error(Aop, approx_op, compute_frobenius=False, row_norms_squared=None):
    """
    Estimate ||A - A_approx|| using Lanczos algorithm (2-norm).
    Optionally compute Frobenius norm using memory-efficient row-by-row computation.
    
    Parameters
    ----------
    Aop : AOperator
        Original matrix operator
    approx_op : object with matvec/lmatvec methods
        Approximation operator (CUR or SVD)
    compute_frobenius : bool
        If True, also compute Frobenius norm using memory-efficient method
    row_norms_squared : jnp.ndarray, optional
        Pre-computed row norms squared for Frobenius computation
        
    Returns
    -------
    error_2norm : float
        Estimated 2-norm (spectral norm) approximation error
    error_frobenius : float, optional
        Frobenius norm approximation error (only if compute_frobenius=True)
    """
    class ErrorOperator:
        def __init__(self, Aop, approx_op):
            self.Aop = Aop
            self.approx_op = approx_op
            self.shape = Aop.shape
            self.dtype = Aop.dtype
        
        def matvec(self, x):
            Ax = self.Aop.matvec(x)
            A_approx_x = self.approx_op.matvec(x)
            return Ax - A_approx_x
        
        def lmatvec(self, y):
            Aty = self.Aop.lmatvec(y)
            A_approx_ty = self.approx_op.lmatvec(y)
            return Aty - A_approx_ty
    
    error_op = ErrorOperator(Aop, approx_op)
    
    # Always use Lanczos for 2-norm estimation
    # Lanczos is memory efficient (O(k) for k iterations) and converges fast
    n, m = error_op.shape
    max_dim = max(n, m)
    
    # Adjust iterations and verbosity based on matrix size
    error_2norm = lanczos_2norm(error_op, max_iter=30, tol=1e-4, random_state=42, verbose= True)
    
    if compute_frobenius:
        # Use memory-efficient Frobenius computation from frobenius_error.py
        # Check if approx_op is a CUR object (CURIncremental or PivotedQRCUR) or SVD
        # Both CURIncremental and PivotedQRCUR have I_array and J_array
        # CURIncremental has U_core_lmatvec, PivotedQRCUR has T_core
        if hasattr(approx_op, 'I_array') and hasattr(approx_op, 'J_array'):
            # CUR object (either CURIncremental or PivotedQRCUR) - use O(n+m) memory method
            # Enable verbose for large matrices to show progress
            verbose_frob = max_dim > 10_000_000
            error_frobenius = compute_frobenius_error_cur(Aop, approx_op, verbose=verbose_frob)
        elif hasattr(approx_op, 'U') and hasattr(approx_op, 's') and (hasattr(approx_op, 'Vt') or hasattr(approx_op, 'Vh')):
            # SVD object - use O(nk+km) memory method
            Vh = approx_op.Vt if hasattr(approx_op, 'Vt') else approx_op.Vh
            error_frobenius = compute_frobenius_error_svd(Aop, approx_op.U, approx_op.s, Vh)
        else:
            raise ValueError(f"Unknown approximation operator: {approx_op}")
        return error_2norm, error_frobenius
    else:
        return error_2norm


def test_comparison(dim=128, ranks=None, problem_type='toeplitz', graph_name='mawi', output_file=None, test_norm_accuracy=False, t_method='svd', skip_svd=False, skip_cur=False, skip_pivoted_qr=False, cur_sampling_options=None, pivoted_qr_cur_sampling_options=None, save_incremental=True, graphs_dir=None, force_frobenius=False, random_seed=42, svd_oversample=5, svd_power_iter=0, svd_batch_matvec=True):
    """
    Compare CURIncremental, PivotedQRCUR, and randomized_svd.
    
    Parameters
    ----------
    dim : int
        Problem size (dim x dim x dim) for Toeplitz, or ignored for graph
    ranks : list of int
        Ranks to test. If None, uses [10, 20, 30, 40, 50]
    cur_sampling_options : list of str, optional
        List of sampling strategies for CURIncremental ('random', 'greedy', 'uniform').
        If None, defaults to ['random']. Default: None
    pivoted_qr_cur_sampling_options : list of str, optional
        List of sampling strategies for pivoted_qr_cur ('random', 'greedy', 'uniform').
        If None, defaults to ['random']. Default: None
    problem_type : str
        Type of problem: 'toeplitz', 'graph', 'line-graph', or 'dense'
    graph_name : str
        Graph name if problem_type='graph': 'mawi', 'snap', or 'web-google'
    t_method : str
        Method for computing T matrix in PivotedQRCUR: 'svd' (use_svd_for_T=True), 
        'pinv' (compute_T_via_pinv=True), or 'direct' (both False). Default: 'svd'
    skip_svd : bool
        If True, skip randomized_svd test entirely. Default: False
    save_incremental : bool
        If True, save results incrementally after each method. Default: True
        
    Returns
    -------
    results : dict
        Dictionary of results for each method
    """
    if ranks is None:
        ranks = [10, 20, 30, 40, 50]
    
    # Convert ranks to list if it's a numpy array
    if isinstance(ranks, np.ndarray):
        ranks = ranks.tolist()
    ranks = sorted(set(ranks))  # Remove duplicates and sort
    
    # Set default sampling options
    if cur_sampling_options is None:
        cur_sampling_options = ['random']
    if pivoted_qr_cur_sampling_options is None:
        pivoted_qr_cur_sampling_options = ['random']
    
    # Ensure sampling options are lists
    if isinstance(cur_sampling_options, str):
        cur_sampling_options = [cur_sampling_options]
    if isinstance(pivoted_qr_cur_sampling_options, str):
        pivoted_qr_cur_sampling_options = [pivoted_qr_cur_sampling_options]
    
    # Sort sampling options so greedy comes first, then random
    cur_sampling_options = sorted(cur_sampling_options, key=lambda x: (x != 'greedy', x))
    pivoted_qr_cur_sampling_options = sorted(pivoted_qr_cur_sampling_options, key=lambda x: (x != 'greedy', x))
    
    print(f"\n{'='*70}")
    print(f"Comparing CURIncremental vs PivotedQRCUR vs randomized_svd")
    print(f"{'='*70}")
    
    # Build operator based on problem type
    if problem_type == 'toeplitz':
        print(f"Problem type: Toeplitz 3D")
        print(f"Problem size: {dim}x{dim}x{dim} = {dim**3:,}")
        print(f"Testing ranks: {ranks}")
        
        kernel_width = 2 * dim + 1
        scale = dim / 320
        kernel = drifted_aniso_gaussian_kernel3d(
            (kernel_width, kernel_width, kernel_width),
            sigmas=np.array([1.0, 1.0, 1.0]) * 80 * scale,
            alpha=1.0,
            beta=0.000,
            delta=50 * scale
        )
        
        Aop = ConvNDOperator(jnp.asarray(kernel, dtype=jnp.float64), (dim, dim, dim), dtype=jnp.float64)
        N = Aop.N
        problem_label = f"Toeplitz {dim}³"
        
        # Compute Frobenius norm if forced or for small problems (N <= 500)
        compute_frobenius = force_frobenius or (N <= 500)
    elif problem_type == 'graph':
        print(f"Problem type: Graph matrix")
        print(f"Graph: {graph_name}")
        print(f"Testing ranks: {ranks}")
        
        Aop, graph_info = load_graph_matrix(graph_name, graphs_dir=graphs_dir)
        N = min(Aop.shape[0], Aop.shape[1])
        problem_label = f"Graph {graph_name} ({Aop.shape[0]}x{Aop.shape[1]})"
        
        # Compute Frobenius norm if forced or for small graphs (N <= 500)
        compute_frobenius = force_frobenius or (N <= 500)
    elif problem_type == 'line-graph':
        print(f"Problem type: Line graph (path graph)")
        print(f"Number of nodes: {dim}")
        print(f"Testing ranks: {ranks}")
        
        # Create line graph adjacency matrix
        A_scipy = create_line_graph_adjacency(dim)
        Aop = SparseMatrixOperator(A_scipy, dtype=jnp.float64)
        N = dim
        problem_label = f"Line Graph ({dim}x{dim})"
        print(f"Matrix shape: {Aop.shape}")
        print(f"Non-zeros: {A_scipy.nnz} (sparsity: {100*(1-A_scipy.nnz/(dim*dim)):.2f}%)")
        
        # Compute Frobenius norm if forced or for small problems (dim <= 500)
        compute_frobenius = force_frobenius or (dim <= 500)
    elif problem_type == 'dense':
        print(f"Problem type: Dense matrix")
        print(f"Matrix size: {dim}x{dim}")
        print(f"Testing ranks: {ranks}")
        
        # Create dense matrix with slowly decaying singular values
        # Use final_ratio=1e-8 for slower decay (was 1e-16)
        A_dense = create_dense_matrix_with_slow_decay(n=dim, random_state=42, final_ratio=1e-16)
        Aop = AOperator(A_dense)
        N = dim
        problem_label = f"Dense {dim}x{dim}"
        print(f"Matrix shape: {Aop.shape}")
        
        # Compute Frobenius norm if forced or for small dense matrices (dim <= 500)
        compute_frobenius = force_frobenius or (dim <= 500)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}. Choose 'toeplitz', 'graph', 'line-graph', or 'dense'")
    
    print(f"Matrix shape: {Aop.shape}")
    print(f"Total elements: {Aop.shape[0] * Aop.shape[1]:,}")
    
    print(f"Matrix shape: {Aop.shape}")
    print(f"Total elements: {N:,}")
    
    # Compute rank 0 error (||A||) using Lanczos algorithm
    print(f"\n  Computing rank 0 error (||A||) via Lanczos...")
    rank_0_error = lanczos_2norm(Aop, max_iter=50, tol=1e-6, random_state=42, verbose=True)
    print(f"  Rank 0 error (2-norm): {rank_0_error:.2e}")
    
    # Compute rank 0 Frobenius norm if needed (||A||_F from row norms)
    rank_0_error_frobenius = None
    row_norms_squared_for_frob = None
    if compute_frobenius:
        # Compute row norms for Frobenius norm (reuse if available)
        if problem_type == 'graph':
            row_norms_squared_for_frob = Aop.compute_all_row_norms_squared()
        else:
            row_norms_squared_for_frob = Aop.compute_all_row_norms_squared()
        rank_0_error_frobenius = float(jnp.sqrt(jnp.sum(row_norms_squared_for_frob)))
        print(f"  Rank 0 error (Frobenius): {rank_0_error_frobenius:.2e}")
    
    # Storage for results - initialize with entries for each sampling combination
    # Each method has started_at, completed_at, and last_updated_at timestamps
    results = {
        'randomized_svd': {'ranks': [0], 'errors': [rank_0_error], 'errors_frobenius': [rank_0_error_frobenius] if compute_frobenius else None, 'times': [0.0], 'started_at': None, 'completed_at': None, 'last_updated_at': None},
    }
    
    # Add entries for each CUR sampling option
    for cur_sampling in cur_sampling_options:
        key = f'cur_incremental_{cur_sampling}'
        results[key] = {'ranks': [0], 'errors': [rank_0_error], 'errors_frobenius': [rank_0_error_frobenius] if compute_frobenius else None, 'times': [0.0], 'iter_when_all_norms_are_negative': -1, 'started_at': None, 'completed_at': None, 'last_updated_at': None}
    
    # Add entries for each pivoted_qr_cur sampling option
    for pqr_sampling in pivoted_qr_cur_sampling_options:
        key = f'pivoted_qr_cur_{pqr_sampling}'
        results[key] = {'ranks': [0], 'errors': [rank_0_error], 'errors_frobenius': [rank_0_error_frobenius] if compute_frobenius else None, 'times': [0.0], 'iter_when_all_norms_are_negative': -1, 'started_at': None, 'completed_at': None, 'last_updated_at': None}
    
    # Create metadata for saving
    metadata = {
        'dim': dim,
        'problem_type': problem_type,
        'graph_name': graph_name if problem_type == 'graph' else None,
        'ranks_tested': ranks,
        'cur_sampling_options': cur_sampling_options,
        'pivoted_qr_cur_sampling_options': pivoted_qr_cur_sampling_options,
        't_method': t_method,
        'skip_svd': skip_svd,
        'rank_0_error': rank_0_error,
        'matrix_shape': list(Aop.shape),
    }
    
    # Define experiment name and plot directory early (used for all outputs)
    if problem_type == 'dense':
        experiment_name = f"dense_{dim}"
    elif problem_type == 'toeplitz':
        experiment_name = f"toeplitz_{dim}"
    elif problem_type == 'line-graph':
        experiment_name = f"linegraph_{dim}"
    elif problem_type == 'graph':
        experiment_name = graph_name  # No _dim suffix for graphs
    else:
        experiment_name = f"{problem_type}_{dim}"
    
    # Determine plot directory:
    # - If output_file is provided, use its directory (consolidate outputs)
    # - Otherwise use default plots directory
    plots_root = os.environ.get("CAUCHY_PLOTS_ROOT", os.path.join(os.getcwd(), "plots"))
    if output_file:
        plot_dir = os.path.dirname(output_file)
        if not plot_dir:  # output_file is just a filename
            plot_dir = os.path.join(plots_root, experiment_name)
    else:
        plot_dir = os.path.join(plots_root, experiment_name)
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # Generate output file if not provided - save in plot_dir
    if output_file is None and save_incremental:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if problem_type == 'toeplitz':
            output_file = f"{plot_dir}/comparison_toeplitz_{dim}_{timestamp}.json"
        elif problem_type == 'graph':
            output_file = f"{plot_dir}/comparison_{graph_name}_{timestamp}.json"
        else:
            output_file = f"{plot_dir}/comparison_{problem_type}_{dim}_{timestamp}.json"
    
    # Helper function for incremental saving
    def save_results_incremental(method_key=None):
        if method_key and method_key in results:
            results[method_key]['last_updated_at'] = datetime.now().isoformat()
        if output_file and save_incremental:
            save_comparison_results(results, metadata, output_file, verbose=False)
    
    # Test CURIncremental for each sampling option
    for cur_sampling in cur_sampling_options:
        print(f"\n{'='*70}")
        if skip_cur:
            print(f"Testing CURIncremental (sampling: {cur_sampling}) - SKIPPED (--skip-cur or --only-pivoted-qr)")
            print(f"{'='*70}")
            continue
        print(f"Testing CURIncremental (sampling: {cur_sampling})")
        print(f"{'='*70}")
        
        result_key = f'cur_incremental_{cur_sampling}'
        results[result_key]['started_at'] = datetime.now().isoformat()
        
        # Build CUR once at max rank
        max_rank = max(r for r in ranks if r <= N)
        print(f"\n  Building CUR at max rank {max_rank}...")
        
        try:
            start_time = time.time()
            if problem_type == 'graph':
                # For sparse matrices, compute row norms squared and take sqrt
                row_norms_squared = Aop.compute_all_row_norms_squared()
            else:
                # For Toeplitz operators
                row_norms_squared = Aop.compute_all_row_norms_squared().block_until_ready()
            
            cur_full = CURIncremental(
                Aop, row_norms_squared,
                debug=False,
                pivot_sampling=cur_sampling,
                use_jit=True,
                store_preconditioner_matrices=False,
                M_inv_matvec=None,
                random_seed=random_seed
            )
            build_result = cur_full.build(max_rank, timing=True)
            
            I_cur = build_result['I'].block_until_ready()
            norm_residuals = build_result['norm_residuals']
            timings_cur = np.array(jax.device_get(build_result['timings']))
            
            # Track when all norms became negative
            iter_neg = build_result.get('iter_when_all_norms_are_negative', -1)
            if iter_neg >= 0:
                results[result_key]['iter_when_all_norms_are_negative'] = iter_neg
                print(f"  CUR: All norms became negative at iteration {iter_neg}")
            
            print(f"  CUR build time: {time.time() - start_time:.3f}s")
            
            # Test at each rank
            for rank in ranks:
                if rank == 0:
                    continue  # Skip rank 0, already added
                if rank > N:
                    continue
                
                print(f"\n  Rank {rank}:")
                try:
                    # Truncate CUR (this includes SVD/pinv computation in truncate_to_rank)
                    truncate_start = time.time()
                    cur_truncated = cur_full.truncate_to_rank(rank)
                    truncate_time = time.time() - truncate_start
                    
                    # Get timing for this rank
                    build_time = float(timings_cur[rank]) if rank < len(timings_cur) else timings_cur[-1]
                    
                    # Estimate error
                    print(f"    Estimating error...")
                    if compute_frobenius:
                        error, error_fro = estimate_approximation_error(Aop, cur_truncated, compute_frobenius=True)
                        print(f"    Error (2-norm): {error:.2e}, Error (Frobenius): {error_fro:.2e}")
                        print(f"    Build time: {build_time:.3f}s")
                        results[result_key]['errors_frobenius'].append(error_fro)
                    else:
                        error = estimate_approximation_error(Aop, cur_truncated, )
                        print(f"    Error: {error:.2e}")
                        print(f"    Build time: {build_time:.3f}s")
                    
                    results[result_key]['ranks'].append(rank)
                    results[result_key]['errors'].append(error)
                    results[result_key]['times'].append(build_time)
                    
                    del cur_truncated
                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()
            
            del cur_full, row_norms_squared
            
            # Clear JAX caches to free memory before pivoted_qr_cur
            safe_clear_jax_caches()
            import gc
            gc.collect()
            
            # Mark method as completed
            results[result_key]['completed_at'] = datetime.now().isoformat()
            save_results_incremental(result_key)
        except Exception as e:
            print(f"  ERROR building CUR: {e}")
            import traceback
            traceback.print_exc()
    
    # Test pivoted_qr_cur for each sampling option
    # Set T computation method based on t_method parameter
    if t_method == 'svd':
        use_svd_for_T = True
        compute_T_via_pinv = False
    elif t_method == 'pinv':
        use_svd_for_T = False
        compute_T_via_pinv = True
    else:  # 'direct'
        use_svd_for_T = False
        compute_T_via_pinv = False
    
    for pqr_sampling in pivoted_qr_cur_sampling_options:
        print(f"\n{'='*70}")
        if skip_pivoted_qr:
            print(f"Testing pivoted_qr_cur (sampling: {pqr_sampling}) - SKIPPED (--skip-pivoted-qr or --only-cur)")
            print(f"{'='*70}")
            continue
        print(f"Testing pivoted_qr_cur (sampling: {pqr_sampling})")
        print(f"{'='*70}")
        
        result_key = f'pivoted_qr_cur_{pqr_sampling}'
        results[result_key]['started_at'] = datetime.now().isoformat()
        
        pivoted_qr_cur_options = {
            'debug': False,
            'column_sampling': pqr_sampling,
            'use_svd_for_T': use_svd_for_T,
            'compute_T_via_pinv': compute_T_via_pinv,
        }
        
        # PivotedQR options (extracted from pivoted_qr_cur_options where relevant)
        pivoted_qr_options = {
            'debug': pivoted_qr_cur_options['debug'],
            'column_sampling': pivoted_qr_cur_options['column_sampling'],
        }
        
        # Pre-compute row and column norms ONCE before testing all ranks
        # This avoids recomputing them for each rank, which causes OOM after CUR
        print(f"\n  Pre-computing row and column norms...")
        row_norms_sq_precomputed = Aop.compute_all_row_norms_squared()
        A_T = Aop.T
        col_norms_sq_precomputed = A_T.compute_all_row_norms_squared()
        del A_T  # Free transpose operator reference
        
        # Filter ranks to compute (exclude rank 0 and ranks > N)
        ranks_to_compute = [r for r in ranks if r > 0 and r <= N]
        
        if ranks_to_compute:
            # Use multi-rank function to compute all ranks efficiently
            print(f"\n  Computing PivotedQRCUR for ranks {ranks_to_compute} (efficient multi-rank)...")
            try:
                multi_rank_results = compute_pivoted_qr_cur_multi_rank(
                    Aop,
                    col_norms_sq_precomputed,
                    row_norms_sq_precomputed,
                    ranks_to_compute,
                    tol=None,  # Use default
                    debug=False,
                    use_jit=True,
                    random_seed=random_seed,  # Use provided seed
                    column_sampling=pivoted_qr_cur_options['column_sampling'],
                    use_svd_for_T=use_svd_for_T,
                    compute_T_via_pinv=compute_T_via_pinv,
                )
                
                # Process results for each rank
                for rank in ranks_to_compute:
                    if rank not in multi_rank_results:
                        print(f"  Skipping rank {rank} (not in results)")
                        continue
                    
                    print(f"\n  Rank {rank}:")
                    try:
                        result = multi_rank_results[rank]
                        cur_obj = result['cur_obj']
                        build_time = result['timings']['total']
                        
                        # Track when all norms became negative (from QR on A and A^T)
                        if cur_obj.qr_result is not None:
                            qr_iter_neg = cur_obj.qr_result.get('iter_when_all_norms_are_negative', -1)
                            if qr_iter_neg >= 0:
                                results[result_key]['iter_when_all_norms_are_negative'] = qr_iter_neg
                                print(f"    PivotedQR (columns): All norms became negative at iteration {qr_iter_neg}")
                        if cur_obj.qr_result_T is not None:
                            qr_T_iter_neg = cur_obj.qr_result_T.get('iter_when_all_norms_are_negative', -1)
                            if qr_T_iter_neg >= 0:
                                # Use the minimum of the two (earliest occurrence)
                                current_neg = results[result_key].get('iter_when_all_norms_are_negative', -1)
                                if current_neg < 0 or qr_T_iter_neg < current_neg:
                                    results[result_key]['iter_when_all_norms_are_negative'] = qr_T_iter_neg
                                print(f"    PivotedQR (rows): All norms became negative at iteration {qr_T_iter_neg}")
                        
                        print(f"    Build time: {build_time:.3f}s")
                        print(f"      - QR cols: {result['timings']['qr_cols']:.3f}s")
                        print(f"      - QR rows: {result['timings']['qr_rows']:.3f}s")
                        print(f"      - Compute T: {result['timings']['compute_T']:.3f}s")
                        
                        # Estimate error
                        print(f"    Estimating error...")
                        if compute_frobenius:
                            error, error_fro = estimate_approximation_error(Aop, cur_obj, compute_frobenius=True)
                            print(f"    Error (2-norm): {error:.2e}, Error (Frobenius): {error_fro:.2e}")
                            results[result_key]['errors_frobenius'].append(error_fro)
                        else:
                            error = estimate_approximation_error(Aop, cur_obj, )
                            print(f"    Error: {error:.2e}")
                        
                        results[result_key]['ranks'].append(rank)
                        results[result_key]['errors'].append(error)
                        results[result_key]['times'].append(build_time)
                        
                        del cur_obj
                        
                        save_results_incremental(result_key)
                    except Exception as e:
                        print(f"    ERROR processing rank {rank}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Clean up
                del multi_rank_results
                
            except Exception as e:
                print(f"  ERROR in multi-rank computation: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to individual rank computation
                print(f"  Falling back to individual rank computation...")
                for rank in ranks_to_compute:
                    if rank == 0:
                        continue
                    if rank > N:
                        print(f"  Skipping rank {rank} (> matrix size {N})")
                        continue
                    
                    print(f"\n  Rank {rank}:")
                    try:
                        start_time = time.time()
                        
                        from pivoted_qr_cur import PivotedQRCUR
                        cur_obj = PivotedQRCUR(
                            Aop, col_norms_sq_precomputed,
                            row_norms_squared=row_norms_sq_precomputed,
                            max_rank=rank,
                            **pivoted_qr_cur_options
                        )
                        cur_obj.build(rank=rank, timing=True)
                        build_time = time.time() - start_time
                        
                        if cur_obj.qr_result is not None:
                            qr_iter_neg = cur_obj.qr_result.get('iter_when_all_norms_are_negative', -1)
                            if qr_iter_neg >= 0:
                                results[result_key]['iter_when_all_norms_are_negative'] = qr_iter_neg
                        if cur_obj.qr_result_T is not None:
                            qr_T_iter_neg = cur_obj.qr_result_T.get('iter_when_all_norms_are_negative', -1)
                            if qr_T_iter_neg >= 0:
                                current_neg = results[result_key].get('iter_when_all_norms_are_negative', -1)
                                if current_neg < 0 or qr_T_iter_neg < current_neg:
                                    results[result_key]['iter_when_all_norms_are_negative'] = qr_T_iter_neg
                        
                        print(f"    Build time: {build_time:.3f}s")
                        
                        print(f"    Estimating error...")
                        if compute_frobenius:
                            error, error_fro = estimate_approximation_error(Aop, cur_obj, compute_frobenius=True)
                            print(f"    Error (2-norm): {error:.2e}, Error (Frobenius): {error_fro:.2e}")
                            results[result_key]['errors_frobenius'].append(error_fro)
                        else:
                            error = estimate_approximation_error(Aop, cur_obj, )
                            print(f"    Error: {error:.2e}")
                        
                        results[result_key]['ranks'].append(rank)
                        results[result_key]['errors'].append(error)
                        results[result_key]['times'].append(build_time)
                        
                        del cur_obj
                        save_results_incremental(result_key)
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Mark method as completed
        results[result_key]['completed_at'] = datetime.now().isoformat()
        save_results_incremental(result_key)
        
        # Clean up pre-computed norms after all ranks tested
        del row_norms_sq_precomputed, col_norms_sq_precomputed
        safe_clear_jax_caches()
        import gc
        gc.collect()
    
    # Test randomized_svd (test last)
    # Skip randomized_svd if explicitly requested or for mawi graph due to memory constraints
    skip_randomized_svd = skip_svd or (problem_type == 'graph' and graph_name == 'mawi')
    
    if skip_randomized_svd:
        if skip_svd:
            print(f"\n{'='*70}")
            print("Testing randomized_svd - SKIPPED (--skip-svd flag)")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print("Testing randomized_svd - SKIPPED (mawi graph, memory constraints)")
            print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("Testing randomized_svd")
        print(f"{'='*70}")
        results['randomized_svd']['started_at'] = datetime.now().isoformat()
    
    svd_failed = False
    for rank in ranks:
        if rank == 0:
            continue  # Skip rank 0, already added
        if rank > N:
            print(f"  Skipping rank {rank} (> matrix size {N})")
            continue
        if svd_failed:
            print(f"  Skipping rank {rank} and larger (randomized_svd ran out of memory)")
            continue
        if skip_randomized_svd:
            print(f"  Skipping rank {rank} (randomized_svd skipped for mawi graph)")
            continue
        
        print(f"\n  Rank {rank}:")
        try:
            start_time = time.time()
            U, s, Vt = randomized_svd(
                Aop,
                n_components=rank,
                n_oversamples=svd_oversample,
                n_iter=svd_power_iter,
                random_state=random_seed,
                batch_matvec=svd_batch_matvec
            )
            build_time = time.time() - start_time
            
            print(f"    Build time: {build_time:.3f}s")
            
            # Create approximation operator
            # A ≈ U @ diag(s) @ Vt, so A^H ≈ Vt^H @ diag(s) @ U^H
            class SVDApprox:
                def __init__(self, U, s, Vt):
                    self.U = U
                    self.s = s
                    self.Vt = Vt  # Keep as Vt for matvec/lmatvec
                    self.shape = (U.shape[0], Vt.shape[1])
                    self.dtype = U.dtype
                
                def matvec(self, x):
                    # A @ x = U @ diag(s) @ Vt @ x
                    return self.U @ (self.s * (self.Vt @ x))
                
                def lmatvec(self, y):
                    # A^H @ y = Vt^H @ diag(s) @ U^H @ y
                    # Vt is (rank, m), so Vt^H is (m, rank)
                    # U is (n, rank), so U^H is (rank, n)
                    return (self.Vt.conj().T) @ (self.s * (self.U.conj().T @ y))
            
            svd_approx = SVDApprox(U, s, Vt)
            
            # Estimate error
            print(f"    Estimating error...")
            if compute_frobenius:
                error, error_fro = estimate_approximation_error(Aop, svd_approx, compute_frobenius=True)
                print(f"    Error (2-norm): {error:.2e}, Error (Frobenius): {error_fro:.2e}")
                results['randomized_svd']['errors_frobenius'].append(error_fro)
            else:
                error = estimate_approximation_error(Aop, svd_approx, )
                print(f"    Error: {error:.2e}")
            
            results['randomized_svd']['ranks'].append(rank)
            results['randomized_svd']['errors'].append(error)
            results['randomized_svd']['times'].append(build_time)
            
            del U, s, Vt, svd_approx
            safe_clear_jax_caches()  # Clear JAX cache after successful SVD
            save_results_incremental('randomized_svd')
        except Exception as e:
            error_msg = str(e).lower()
            # Check for OOM errors (JAX, CUDA, or general memory errors)
            is_oom = (
                'memory' in error_msg or 
                'out of memory' in error_msg or 
                'cuda' in error_msg or 
                'oom' in error_msg or
                'resource_exhausted' in error_msg or
                isinstance(e, MemoryError) or
                (hasattr(jax.errors, 'JaxRuntimeError') and isinstance(e, jax.errors.JaxRuntimeError) and 'resource' in error_msg)
            )
            
            if is_oom:
                print(f"    ERROR: Out of memory - stopping randomized_svd for larger ranks")
                print(f"    (This is expected for large problems - continuing with CUR and pivoted_qr_cur)")
                svd_failed = True
                safe_clear_jax_caches()  # Clear cache after OOM
            else:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                safe_clear_jax_caches()
    
    # Mark SVD as completed (if it wasn't skipped)
    if not skip_randomized_svd:
        results['randomized_svd']['completed_at'] = datetime.now().isoformat()
        save_results_incremental('randomized_svd')
    
    # Norm estimation accuracy comparison (only for dense matrices)
    if problem_type == 'dense' and test_norm_accuracy:
        print(f"\n{'='*70}")
        print("Norm Estimation Accuracy Comparison")
        print(f"{'='*70}")
        
        # Use the main plot_dir already defined earlier (all outputs in same place)
        # For norm accuracy sub-experiments, save to same plot_dir
        
        # Test at a finer grid of ranks
        # Use every 25 ranks up to 500, then every 50 up to N
        suggested_test_ranks = list(range(25, 501, 25)) + list(range(550, min(N, 1001), 50))
        test_ranks = [r for r in suggested_test_ranks if r in ranks and r <= N]
        # If no ranks match, use a subset of the provided ranks
        if not test_ranks:
            test_ranks = [r for r in ranks if r > 0 and r <= N]
            # Take every other rank if there are many
            if len(test_ranks) > 20:
                test_ranks = test_ranks[::len(test_ranks)//20]
        
        # Ensure we have at least some ranks to test
        if not test_ranks:
            print("WARNING: No valid ranks found for norm accuracy testing. Skipping norm accuracy tests.")
            test_norm_accuracy = False
        else:
            print(f"Testing norm accuracy at ranks: {test_ranks}")
        
        norm_accuracy_results = {
            'pivoted_qr_cur': {
                'ranks': [], 
                'col_norm_rel_errors': [], 
                'col_norm_abs_errors': [],
                'row_norm_rel_errors': [],  # Row norms from QR on A^T
                'row_norm_abs_errors': [],
                'all_true_col_norms': {},  # rank -> array of true norms for all remaining cols
                'all_estimated_col_norms': {},  # rank -> array of estimated norms for all remaining cols
                'all_true_row_norms': {},  # rank -> array of true norms for all remaining rows (from A^T)
                'all_estimated_row_norms': {}  # rank -> array of estimated norms for all remaining rows (from A^T)
            },
            'cur_incremental': {
                'ranks': [], 
                'row_norm_rel_errors': [], 
                'row_norm_abs_errors': [],
                'all_true_row_norms': {},  # rank -> array of true norms for all remaining rows
                'all_estimated_row_norms': {}  # rank -> array of estimated norms for all remaining rows
            }
        }
        
        # Test pivoted_qr_cur column norm accuracy
        print("\nTesting pivoted_qr_cur column norm estimation accuracy...")
        
        # Ensure PivotedQR is available (import at top level, but ensure it's accessible)
        from pivoted_qr import PivotedQR
        
        for rank in test_ranks:
            print(f"\n  Rank {rank}:")
            try:
                # Build QR up to this rank
                row_norms_sq = Aop.compute_all_row_norms_squared()
                A_T = Aop.T
                col_norms_sq = A_T.compute_all_row_norms_squared()
                
                qr_obj = PivotedQR(Aop, col_norms_sq.copy(), max_rank=rank, tol=None, **pivoted_qr_options)
                for step in range(rank):
                    qr_obj.step()
                
                # Get estimated column norms for remaining columns
                remaining_cols_mask = qr_obj.remaining_cols_mask
                remaining_col_indices = np.where(remaining_cols_mask)[0]
                estimated_col_norms_sq = np.asarray(qr_obj.col_norms_sq[remaining_cols_mask])
                
                # Get selected columns
                selected_cols = np.asarray(qr_obj.chosen_columns[:qr_obj.k])
                selected_cols = selected_cols[(selected_cols >= 0) & (selected_cols < N)]
                
                # Extract full dense matrix A (for dense problem type)
                if problem_type == 'dense':
                    A_dense = np.asarray(Aop.A)  # AOperator stores A directly
                else:
                    # For other problem types, extract columns one by one
                    A_dense = np.zeros((Aop.shape[0], Aop.shape[1]), dtype=Aop.dtype)
                    for j in range(Aop.shape[1]):
                        e_j = np.zeros(Aop.shape[1])
                        e_j[j] = 1.0
                        A_dense[:, j] = np.asarray(Aop.matvec(jnp.asarray(e_j)))
                
                # Compute dense QR of chosen columns: A[:, selected_cols] = Q @ R
                A_selected = A_dense[:, selected_cols]
                Q, R_dense = qr(A_selected, mode='economic')  # Q is (N, rank), R is (rank, rank)
                
                # Compute R_2 = Q^T @ A (this is the projection of all columns onto Q)
                R_2 = Q.T @ A_dense  # Shape: (rank, N)
                
                # Compute column norms of residual efficiently without forming full residual matrix
                # ||A[:,j] - Q @ R_2[:,j]||^2 = ||A[:,j]||^2 - 2*(A[:,j]^T @ Q @ R_2[:,j]) + ||Q @ R_2[:,j]||^2
                # But simpler: compute column by column or use vectorized operations
                # For efficiency, compute A_approx columns on the fly
                A_approx_cols = Q @ R_2  # Shape: (N, N) - but we can compute norms column-wise
                # Actually, we can compute ||A - Q@R_2||_F^2 per column more efficiently:
                # ||A[:,j] - (Q@R_2)[:,j]||^2 = ||A[:,j]||^2 - 2*A[:,j]^T @ (Q@R_2)[:,j] + ||(Q@R_2)[:,j]||^2
                # But for simplicity and correctness, compute full residual (it's only 1000x1000)
                residual = A_dense - (Q @ R_2)  # Shape: (N, N)
                
                # Compute column norms of residual (true column error norms)
                true_col_error_norms_sq = np.sum(residual**2, axis=0)  # Shape: (N,)
                
                # Process ALL remaining columns (not just a sample)
                remaining_col_indices = remaining_col_indices[(remaining_col_indices >= 0) & (remaining_col_indices < N)]
                
                # Get true and estimated norms for all remaining columns
                true_col_errors = []
                estimated_col_errors = []
                
                for col_idx in remaining_col_indices:
                    # Get estimated norm
                    col_pos = np.where(remaining_col_indices == col_idx)[0]
                    if len(col_pos) == 0:
                        continue
                    estimated_norm_sq = float(estimated_col_norms_sq[col_pos[0]])
                    estimated_col_errors.append(estimated_norm_sq)
                    
                    # Get TRUE column error norm from residual
                    true_error_norm_sq = float(true_col_error_norms_sq[col_idx])
                    true_col_errors.append(true_error_norm_sq)
                
                if true_col_errors and estimated_col_errors:
                    true_array = np.array(true_col_errors)
                    estimated_array = np.array(estimated_col_errors)
                    
                    # Relative and absolute errors
                    denominator = np.maximum(true_array, np.maximum(estimated_array, 1e-20))
                    rel_errors = np.abs(estimated_array - true_array) / denominator
                    abs_errors = np.abs(estimated_array - true_array)
                    
                    norm_accuracy_results['pivoted_qr_cur']['ranks'].append(rank)
                    norm_accuracy_results['pivoted_qr_cur']['col_norm_rel_errors'].append(float(np.mean(rel_errors)))
                    norm_accuracy_results['pivoted_qr_cur']['col_norm_abs_errors'].append(float(np.mean(abs_errors)))
                    
                    # Store full arrays for plotting
                    norm_accuracy_results['pivoted_qr_cur']['all_true_col_norms'][rank] = true_array
                    norm_accuracy_results['pivoted_qr_cur']['all_estimated_col_norms'][rank] = estimated_array
                    
                    print(f"    Mean rel error: {np.mean(rel_errors):.2%}, Mean abs error: {np.mean(abs_errors):.2e}, "
                          f"Max rel error: {np.max(rel_errors):.2%}, Num cols: {len(true_col_errors)}")
                
                del qr_obj
            except Exception as e:
                print(f"    ERROR: {e}")
        
        # Test pivoted_qr_cur row norm accuracy (by running QR on A^T)
        print("\nTesting pivoted_qr_cur row norm estimation accuracy (via A^T)...")
        
        # Ensure PivotedQR is available (import at top level, but ensure it's accessible)
        from pivoted_qr import PivotedQR
        
        for rank in test_ranks:
            print(f"\n  Rank {rank}:")
            try:
                # Build QR on A^T up to this rank
                # A^T has shape (N, N), so columns of A^T are rows of A
                # PivotedQR needs column norms of the matrix it operates on
                # Column norms of A^T = row norms of A
                A_T = Aop.T
                # Compute column norms of A^T (which are row norms of A)
                col_norms_sq_A_T = Aop.compute_all_row_norms_squared()  # Row norms of A = column norms of A^T
                
                qr_obj_A_T = PivotedQR(A_T, col_norms_sq_A_T.copy(), max_rank=rank, tol=None, **pivoted_qr_options)
                for step in range(rank):
                    qr_obj_A_T.step()
                
                # Get estimated row norms for remaining rows (columns of A^T)
                remaining_rows_mask = qr_obj_A_T.remaining_cols_mask
                remaining_row_indices = np.where(remaining_rows_mask)[0]
                estimated_row_norms_sq = np.asarray(qr_obj_A_T.col_norms_sq[remaining_rows_mask])
                
                # Get selected rows (columns of A^T)
                selected_rows = np.asarray(qr_obj_A_T.chosen_columns[:qr_obj_A_T.k])
                selected_rows = selected_rows[(selected_rows >= 0) & (selected_rows < N)]
                
                # Extract full dense matrix A (for dense problem type)
                if problem_type == 'dense':
                    A_dense = np.asarray(Aop.A)  # AOperator stores A directly
                else:
                    # For other problem types, extract matrix
                    A_dense = np.zeros((Aop.shape[0], Aop.shape[1]), dtype=Aop.dtype)
                    for j in range(Aop.shape[1]):
                        e_j = np.zeros(Aop.shape[1])
                        e_j[j] = 1.0
                        A_dense[:, j] = np.asarray(Aop.matvec(jnp.asarray(e_j)))
                
                # Compute dense QR of chosen rows: A^T[:, selected_rows] = Q @ R
                # This is equivalent to QR on rows of A
                A_T_dense = A_dense.T
                A_T_selected = A_T_dense[:, selected_rows]
                Q_T, R_T_dense = qr(A_T_selected, mode='economic')  # Q_T is (N, rank), R_T is (rank, rank)
                
                # Compute R_2 = Q_T^T @ A^T (this is the projection of all rows onto Q_T)
                R_2_T = Q_T.T @ A_T_dense  # Shape: (rank, N)
                
                # Compute residual: A^T - Q_T @ R_2_T
                residual_T = A_T_dense - (Q_T @ R_2_T)  # Shape: (N, N)
                
                # Compute column norms of residual (true row error norms of A)
                true_row_error_norms_sq = np.sum(residual_T**2, axis=0)  # Shape: (N,)
                
                # Process ALL remaining rows (not just a sample)
                remaining_row_indices = remaining_row_indices[(remaining_row_indices >= 0) & (remaining_row_indices < N)]
                
                # Get true and estimated norms for all remaining rows
                true_row_errors = []
                estimated_row_errors = []
                
                for row_idx in remaining_row_indices:
                    # Get estimated norm
                    row_pos = np.where(remaining_row_indices == row_idx)[0]
                    if len(row_pos) == 0:
                        continue
                    estimated_norm_sq = float(estimated_row_norms_sq[row_pos[0]])
                    estimated_row_errors.append(estimated_norm_sq)
                    
                    # Get TRUE row error norm from residual
                    true_error_norm_sq = float(true_row_error_norms_sq[row_idx])
                    true_row_errors.append(true_error_norm_sq)
                
                if true_row_errors and estimated_row_errors:
                    true_array = np.array(true_row_errors)
                    estimated_array = np.array(estimated_row_errors)
                    
                    # Relative and absolute errors
                    denominator = np.maximum(true_array, np.maximum(estimated_array, 1e-20))
                    rel_errors = np.abs(estimated_array - true_array) / denominator
                    abs_errors = np.abs(estimated_array - true_array)
                    
                    # Store row norm results (ranks should match column norm ranks since we use same test_ranks)
                    norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors'].append(float(np.mean(rel_errors)))
                    norm_accuracy_results['pivoted_qr_cur']['row_norm_abs_errors'].append(float(np.mean(abs_errors)))
                    
                    # Store full arrays for plotting
                    norm_accuracy_results['pivoted_qr_cur']['all_true_row_norms'][rank] = true_array
                    norm_accuracy_results['pivoted_qr_cur']['all_estimated_row_norms'][rank] = estimated_array
                    
                    print(f"    Mean rel error: {np.mean(rel_errors):.2%}, Mean abs error: {np.mean(abs_errors):.2e}, "
                          f"Max rel error: {np.max(rel_errors):.2%}, Num rows: {len(true_row_errors)}")
                
                del qr_obj_A_T
            except Exception as e:
                print(f"    ERROR: {e}")
        
        # Test CURIncremental row norm accuracy
        print("\nTesting CURIncremental row norm estimation accuracy...")
        
        for rank in test_ranks:
            print(f"\n  Rank {rank}:")
            try:
                # Build CUR up to this rank
                row_norms_sq = Aop.compute_all_row_norms_squared()
                cur_obj = CURIncremental(
                    Aop, row_norms_sq,
                    debug=False,
                    pivot_sampling='greedy',
                    store_preconditioner_matrices=False,
                    use_jit=True
                )
                for step in range(rank):
                    cur_obj.step()
                
                # Get selected rows and columns
                I_selected = np.asarray(cur_obj.I_array[:cur_obj.k])
                J_selected = np.asarray(cur_obj.J_array[:cur_obj.k])
                I_selected = I_selected[(I_selected >= 0) & (I_selected < N)]
                J_selected = J_selected[(J_selected >= 0) & (J_selected < N)]
                
                # Extract full dense matrix A (for dense problem type)
                if problem_type == 'dense':
                    A_dense = np.asarray(Aop.A)  # AOperator stores A directly
                else:
                    # For other problem types, extract matrix
                    A_dense = np.zeros((Aop.shape[0], Aop.shape[1]), dtype=Aop.dtype)
                    for j in range(Aop.shape[1]):
                        e_j = np.zeros(Aop.shape[1])
                        e_j[j] = 1.0
                        A_dense[:, j] = np.asarray(Aop.matvec(jnp.asarray(e_j)))
                
                # Extract A[I, J] (intersection of selected rows and columns)
                A_IJ = A_dense[np.ix_(I_selected, J_selected)]  # Shape: (rank, rank)
                
                # Compute U = pinv(A[I, J])
                U_core = np.linalg.pinv(A_IJ)  # Shape: (rank, rank)
                
                # Extract C = A[:, J] and R = A[I, :]
                C = A_dense[:, J_selected]  # Shape: (N, rank)
                R = A_dense[I_selected, :]  # Shape: (rank, N)
                
                # Compute CUR approximation: A_approx = C @ U @ R
                A_approx = C @ U_core @ R  # Shape: (N, N)
                
                # Compute residual: A - A_approx
                residual = A_dense - A_approx  # Shape: (N, N)
                
                # Compute row norms of residual (true row error norms)
                true_row_error_norms_sq = np.sum(residual**2, axis=1)  # Shape: (N,)
                
                # Get estimated row norms
                estimated_row_norms_sq = np.asarray(cur_obj.n_row)
                
                # Process ALL remaining rows (not just a sample)
                # Get all rows that are not selected
                all_rows = np.arange(N)
                remaining_rows = np.setdiff1d(all_rows, I_selected)
                
                true_row_errors = []
                estimated_row_errors = []
                
                for i in remaining_rows:
                    # Get estimated norm
                    estimated_norm_sq = float(estimated_row_norms_sq[i])
                    estimated_row_errors.append(estimated_norm_sq)
                    
                    # Get TRUE row error norm from residual
                    true_error_norm_sq = float(true_row_error_norms_sq[i])
                    true_row_errors.append(true_error_norm_sq)
                
                if true_row_errors and estimated_row_errors:
                    true_array = np.array(true_row_errors)
                    estimated_array = np.array(estimated_row_errors)
                    
                    # Relative and absolute errors
                    denominator = np.maximum(true_array, np.maximum(estimated_array, 1e-20))
                    rel_errors = np.abs(estimated_array - true_array) / denominator
                    abs_errors = np.abs(estimated_array - true_array)
                    
                    norm_accuracy_results['cur_incremental']['ranks'].append(rank)
                    norm_accuracy_results['cur_incremental']['row_norm_rel_errors'].append(float(np.mean(rel_errors)))
                    norm_accuracy_results['cur_incremental']['row_norm_abs_errors'].append(float(np.mean(abs_errors)))
                    
                    # Store full arrays for plotting
                    norm_accuracy_results['cur_incremental']['all_true_row_norms'][rank] = true_array
                    norm_accuracy_results['cur_incremental']['all_estimated_row_norms'][rank] = estimated_array
                    
                    print(f"    Mean rel error: {np.mean(rel_errors):.2%}, Mean abs error: {np.mean(abs_errors):.2e}, "
                          f"Max rel error: {np.max(rel_errors):.2%}, Num rows: {len(true_row_errors)}")
                
                del cur_obj
            except Exception as e:
                print(f"    ERROR: {e}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("Norm Estimation Accuracy Summary")
        print(f"{'='*70}")
        print(f"\n{'Rank':<8} {'QR Col Rel Err':<18} {'QR Row Rel Err':<18} {'CUR Row Rel Err':<18}")
        print("-" * 80)
        
        # Find common ranks
        qr_ranks = norm_accuracy_results['pivoted_qr_cur']['ranks']
        cur_ranks = norm_accuracy_results['cur_incremental']['ranks']
        all_ranks = sorted(set(qr_ranks) & set(cur_ranks))
        
        for rank in all_ranks:
            qr_idx = qr_ranks.index(rank)
            cur_idx = cur_ranks.index(rank)
            qr_col_err = norm_accuracy_results['pivoted_qr_cur']['col_norm_rel_errors'][qr_idx]
            qr_row_err = norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors'][qr_idx] if qr_idx < len(norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors']) else float('nan')
            cur_row_err = norm_accuracy_results['cur_incremental']['row_norm_rel_errors'][cur_idx]
            print(f"{rank:<8} "
                  f"{qr_col_err:<18.2%} "
                  f"{qr_row_err:<18.2%} "
                  f"{cur_row_err:<18.2%}")
        
        # Create norm accuracy plots
        # Check if we have any results (either column or row norms)
        has_results = (norm_accuracy_results['pivoted_qr_cur']['ranks'] or 
                      norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors'] or
                      norm_accuracy_results['cur_incremental']['ranks'])
        if problem_type == 'dense' and has_results:
            print(f"\n{'='*70}")
            print("Creating norm accuracy plots")
            print(f"{'='*70}")
            
            # Find common ranks for plotting
            qr_ranks_plot = norm_accuracy_results['pivoted_qr_cur']['ranks']
            cur_ranks_plot = norm_accuracy_results['cur_incremental']['ranks']
            common_ranks = sorted(set(qr_ranks_plot) & set(cur_ranks_plot))
            
            # Plot 1: Mean relative error vs rank (all on same plot with same scale)
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            
            # Collect all data to determine common scale
            all_rel_errors = []
            
            # QR column norm accuracy
            qr_col_rel_errors = norm_accuracy_results['pivoted_qr_cur']['col_norm_rel_errors']
            if qr_col_rel_errors and len(qr_col_rel_errors) == len(qr_ranks_plot):
                # Convert to numpy array and handle zeros for log scale
                qr_col_array = np.array(qr_col_rel_errors)
                # Replace zeros with a small value for log scale plotting
                qr_col_array = np.maximum(qr_col_array, 1e-8)
                all_rel_errors.extend(qr_col_array)
                ax.semilogy(qr_ranks_plot, qr_col_array, 'o-', color='#9467bd', linewidth=2, markersize=8, label='QR Column Norms', alpha=0.8)
                print(f"  QR Column Norms: {len(qr_col_rel_errors)} values, range: [{np.min(qr_col_array):.2e}, {np.max(qr_col_array):.2e}]")
            
            # QR row norm accuracy
            if norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors']:
                qr_row_rel_errors = norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors']
                # QR row norms should be computed for same ranks as column norms
                if len(qr_row_rel_errors) == len(qr_ranks_plot):
                    qr_row_array = np.array(qr_row_rel_errors)
                    qr_row_array = np.maximum(qr_row_array, 1e-8)
                    all_rel_errors.extend(qr_row_array)
                    ax.semilogy(qr_ranks_plot, qr_row_array, '^-', color='#8c564b', linewidth=2, markersize=8, label='QR Row Norms (A^T)', alpha=0.8)
                    print(f"  QR Row Norms: {len(qr_row_rel_errors)} values, range: [{np.min(qr_row_array):.2e}, {np.max(qr_row_array):.2e}]")
            
            # CUR row norm accuracy
            cur_rel_errors = norm_accuracy_results['cur_incremental']['row_norm_rel_errors']
            if cur_rel_errors and len(cur_rel_errors) == len(cur_ranks_plot):
                cur_array = np.array(cur_rel_errors)
                cur_array = np.maximum(cur_array, 1e-8)
                all_rel_errors.extend(cur_array)
                ax.semilogy(cur_ranks_plot, cur_array, 's-', color='#2ca02c', linewidth=2, markersize=8, label='CUR Row Norms', alpha=0.8)
                print(f"  CUR Row Norms: {len(cur_rel_errors)} values, range: [{np.min(cur_array):.2e}, {np.max(cur_array):.2e}]")
            
            # Set better y-axis limits for readability
            if all_rel_errors:
                all_rel_errors_array = np.array(all_rel_errors)
                # Use percentiles to avoid outliers dominating the scale
                y_min = np.percentile(all_rel_errors_array, 1)  # 1st percentile
                y_max = np.percentile(all_rel_errors_array, 99)  # 99th percentile
                # Add some padding
                y_min = max(1e-8, y_min * 0.3)
                y_max = y_max * 3
                # But ensure we show at least the full range of each method
                if qr_col_rel_errors and len(qr_col_rel_errors) == len(qr_ranks_plot):
                    qr_col_min = np.percentile(qr_col_array, 1)
                    qr_col_max = np.percentile(qr_col_array, 99)
                    y_min = min(y_min, max(1e-8, qr_col_min * 0.3))
                    y_max = max(y_max, qr_col_max * 3)
                if norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors'] and len(qr_row_rel_errors) == len(qr_ranks_plot):
                    qr_row_min = np.percentile(qr_row_array, 1)
                    qr_row_max = np.percentile(qr_row_array, 99)
                    y_min = min(y_min, max(1e-8, qr_row_min * 0.3))
                    y_max = max(y_max, qr_row_max * 3)
                if cur_rel_errors and len(cur_rel_errors) == len(cur_ranks_plot):
                    cur_min = np.percentile(cur_array, 1)
                    cur_max = np.percentile(cur_array, 99)
                    y_min = min(y_min, max(1e-8, cur_min * 0.3))
                    y_max = max(y_max, cur_max * 3)
                ax.set_ylim(y_min, y_max)
                print(f"  Plot y-axis range: [{y_min:.2e}, {y_max:.2e}]")
            
            ax.set_xlabel('Rank', fontsize=12)
            ax.set_ylabel('Mean Relative Error', fontsize=12)
            ax.set_title('Norm Estimation Accuracy Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11, loc='best')
            
            plt.tight_layout()
            plot_filename = f'{plot_dir}/norm_accuracy_vs_rank.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {plot_filename}")
            plt.close()
            
            # Save results to JSON file
            results_dict = {
                'experiment': experiment_name,
                'dimension': dim,
                'problem_type': problem_type,
                'ranks': all_ranks,
                'qr_column_norms': {
                    'ranks': qr_ranks_plot.tolist() if hasattr(qr_ranks_plot, 'tolist') else list(qr_ranks_plot),
                    'relative_errors': [float(x) for x in qr_col_rel_errors] if qr_col_rel_errors else [],
                    'absolute_errors': norm_accuracy_results['pivoted_qr_cur']['col_norm_abs_errors']
                },
                'qr_row_norms': {
                    'ranks': qr_ranks_plot.tolist() if hasattr(qr_ranks_plot, 'tolist') else list(qr_ranks_plot),
                    'relative_errors': [float(x) for x in qr_row_rel_errors] if norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors'] and len(norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors']) == len(qr_ranks_plot) else [],
                    'absolute_errors': norm_accuracy_results['pivoted_qr_cur']['row_norm_abs_errors']
                },
                'cur_row_norms': {
                    'ranks': cur_ranks_plot.tolist() if hasattr(cur_ranks_plot, 'tolist') else list(cur_ranks_plot),
                    'relative_errors': [float(x) for x in cur_rel_errors] if cur_rel_errors else [],
                    'absolute_errors': norm_accuracy_results['cur_incremental']['row_norm_abs_errors']
                }
            }
            json_filename = f'{plot_dir}/norm_accuracy_results.json'
            with open(json_filename, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Saved: {json_filename}")
            
            # Also save as readable text file
            txt_filename = f'{plot_dir}/norm_accuracy_results.txt'
            with open(txt_filename, 'w') as f:
                f.write(f"Norm Estimation Accuracy Results\n")
                f.write(f"{'='*80}\n")
                f.write(f"Experiment: {experiment_name}\n")
                f.write(f"Dimension: {dim}\n")
                f.write(f"Problem Type: {problem_type}\n")
                f.write(f"\n{'Rank':<8} {'QR Col Rel Err':<20} {'QR Row Rel Err':<20} {'CUR Row Rel Err':<20}\n")
                f.write("-" * 80 + "\n")
                for rank in all_ranks:
                    qr_idx = qr_ranks.index(rank)
                    cur_idx = cur_ranks.index(rank)
                    qr_col_err = norm_accuracy_results['pivoted_qr_cur']['col_norm_rel_errors'][qr_idx]
                    qr_row_err = norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors'][qr_idx] if qr_idx < len(norm_accuracy_results['pivoted_qr_cur']['row_norm_rel_errors']) else float('nan')
                    cur_row_err = norm_accuracy_results['cur_incremental']['row_norm_rel_errors'][cur_idx]
                    f.write(f"{rank:<8} {qr_col_err:<20.6%} {qr_row_err:<20.6%} {cur_row_err:<20.6%}\n")
            print(f"Saved: {txt_filename}")
            
            # Plot 2: Scatter plots of true vs estimated norms for selected ranks
            # Include QR row norms in common ranks
            all_plot_ranks = sorted(set(common_ranks) | set(norm_accuracy_results['pivoted_qr_cur']['all_true_row_norms'].keys()))
            selected_plot_ranks = sorted(all_plot_ranks)[::max(1, len(all_plot_ranks)//4)]  # Plot ~4 ranks
            if selected_plot_ranks:
                n_ranks = len(selected_plot_ranks)
                fig, axes = plt.subplots(3, n_ranks, figsize=(6*n_ranks, 15))
                if n_ranks == 1:
                    axes = axes.reshape(3, 1)
                
                for idx, rank in enumerate(selected_plot_ranks):
                    # QR column scatter plot
                    if rank in norm_accuracy_results['pivoted_qr_cur']['all_true_col_norms']:
                        true_qr = norm_accuracy_results['pivoted_qr_cur']['all_true_col_norms'][rank]
                        est_qr = norm_accuracy_results['pivoted_qr_cur']['all_estimated_col_norms'][rank]
                        axes[0, idx].loglog(true_qr, est_qr, 'o', alpha=0.5, color='#9467bd', markersize=3)
                        # Perfect line
                        min_val = min(np.min(true_qr), np.min(est_qr))
                        max_val = max(np.max(true_qr), np.max(est_qr))
                        axes[0, idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect')
                        axes[0, idx].set_xlabel('True Column Norm²', fontsize=10)
                        axes[0, idx].set_ylabel('Estimated Column Norm²', fontsize=10)
                        axes[0, idx].set_title(f'QR Columns Rank {rank}', fontsize=11, fontweight='bold')
                        axes[0, idx].legend(fontsize=9)
                        axes[0, idx].grid(True, alpha=0.3)
                    
                    # QR row scatter plot (from A^T)
                    if rank in norm_accuracy_results['pivoted_qr_cur']['all_true_row_norms']:
                        true_qr_row = norm_accuracy_results['pivoted_qr_cur']['all_true_row_norms'][rank]
                        est_qr_row = norm_accuracy_results['pivoted_qr_cur']['all_estimated_row_norms'][rank]
                        axes[1, idx].loglog(true_qr_row, est_qr_row, '^', alpha=0.5, color='#8c564b', markersize=3)
                        # Perfect line
                        min_val = min(np.min(true_qr_row), np.min(est_qr_row))
                        max_val = max(np.max(true_qr_row), np.max(est_qr_row))
                        axes[1, idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect')
                        axes[1, idx].set_xlabel('True Row Norm²', fontsize=10)
                        axes[1, idx].set_ylabel('Estimated Row Norm²', fontsize=10)
                        axes[1, idx].set_title(f'QR Rows (A^T) Rank {rank}', fontsize=11, fontweight='bold')
                        axes[1, idx].legend(fontsize=9)
                        axes[1, idx].grid(True, alpha=0.3)
                    
                    # CUR scatter plot
                    if rank in norm_accuracy_results['cur_incremental']['all_true_row_norms']:
                        true_cur = norm_accuracy_results['cur_incremental']['all_true_row_norms'][rank]
                        est_cur = norm_accuracy_results['cur_incremental']['all_estimated_row_norms'][rank]
                        axes[2, idx].loglog(true_cur, est_cur, 's', alpha=0.5, color='#2ca02c', markersize=3)
                        # Perfect line
                        min_val = min(np.min(true_cur), np.min(est_cur))
                        max_val = max(np.max(true_cur), np.max(est_cur))
                        axes[2, idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect')
                        axes[2, idx].set_xlabel('True Row Norm²', fontsize=10)
                        axes[2, idx].set_ylabel('Estimated Row Norm²', fontsize=10)
                        axes[2, idx].set_title(f'CUR Rows Rank {rank}', fontsize=11, fontweight='bold')
                        axes[2, idx].legend(fontsize=9)
                        axes[2, idx].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_filename = f'{plot_dir}/norm_scatter.png'
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"Saved: {plot_filename}")
                plt.close()
        
        # R matrix comparison (only for dense matrices, and only if norm accuracy is enabled)
        if test_norm_accuracy:
            print(f"\n{'='*70}")
            print("R Matrix Comparison (QR columns)")
            print(f"{'='*70}")
            
            r_comparison_results = {
                'ranks': [],
                'r_cols_errors': [],  # Errors in R_cols (from QR on columns)
                'r_rows_errors': []   # Errors in R_rows (from QR on rows via A^T)
            }
            
            # Use same test ranks as norm accuracy
            for rank in test_ranks:
                print(f"\n  Rank {rank}:")
                try:
                    # Build PivotedQRCUR to get R matrices
                    row_norms_sq = Aop.compute_all_row_norms_squared()
                    A_T = Aop.T
                    col_norms_sq = A_T.compute_all_row_norms_squared()
                    
                    from pivoted_qr_cur import PivotedQRCUR
                    cur_obj = PivotedQRCUR(
                        Aop, col_norms_sq,
                        row_norms_squared=row_norms_sq,
                        max_rank=rank,
                        **pivoted_qr_cur_options
                    )
                    cur_obj.build(rank=rank)
                    
                    # Extract R_cols from pivoted_qr_cur
                    R_cols_pivoted = np.asarray(cur_obj.qr_result['R'])  # k x k
                    selected_cols = np.asarray(cur_obj.qr_result['I'][:rank])
                    selected_cols = selected_cols[(selected_cols >= 0) & (selected_cols < N)]
                    
                    # Extract full dense matrix A
                    if problem_type == 'dense':
                        A_dense = np.asarray(Aop.A)
                    else:
                        A_dense = np.zeros((Aop.shape[0], Aop.shape[1]), dtype=Aop.dtype)
                        for j in range(Aop.shape[1]):
                            e_j = np.zeros(Aop.shape[1])
                            e_j[j] = 1.0
                            A_dense[:, j] = np.asarray(Aop.matvec(jnp.asarray(e_j)))
                    
                    # Compute scipy QR of selected columns: A[:, selected_cols] = Q @ R
                    A_selected = A_dense[:, selected_cols]
                    Q_scipy, R_scipy = qr(A_selected, mode='economic')  # Q is (N, rank), R is (rank, rank)
                    
                    # Scale R_scipy rows so that r_ii > 0
                    # If r_ii < 0, multiply row i by -1
                    for i in range(R_scipy.shape[0]):
                        if R_scipy[i, i] < 0:
                            R_scipy[i, :] *= -1
                            # Also need to adjust Q to maintain A = Q @ R
                            Q_scipy[:, i] *= -1
                    
                    # Compare R_cols: compute relative error
                    # Normalize by the norm of R_scipy for relative error
                    R_scipy_norm = np.linalg.norm(R_scipy, 'fro')
                    R_pivoted_norm = np.linalg.norm(R_cols_pivoted, 'fro')
                    
                    if R_scipy_norm > 0:
                        r_cols_error = np.linalg.norm(R_cols_pivoted - R_scipy, 'fro') / R_scipy_norm
                    else:
                        r_cols_error = np.linalg.norm(R_cols_pivoted - R_scipy, 'fro')
                    
                    r_comparison_results['ranks'].append(rank)
                    r_comparison_results['r_cols_errors'].append(float(r_cols_error))
                    
                    print(f"    R_cols relative error: {r_cols_error:.2e}")
                    
                    # Also compare R_rows (from QR on A^T)
                    # Extract R_rows from pivoted_qr_cur (stored in qr_result_T)
                    if cur_obj.qr_result_T is not None:
                        R_rows_pivoted = np.asarray(cur_obj.qr_result_T['R'])  # k x k
                        selected_rows = np.asarray(cur_obj.qr_result_T['I'][:rank])
                    else:
                        # Fallback: recompute if not stored
                        A_T = Aop.T
                        col_norms_sq_A_T = Aop.compute_all_row_norms_squared()  # Row norms of A = column norms of A^T
                        
                        # Use top-level import of PivotedQR (already imported at module level)
                        qr_obj_A_T = PivotedQR(A_T, col_norms_sq_A_T.copy(), max_rank=rank, tol=None, **pivoted_qr_options)
                        for step in range(rank):
                            qr_obj_A_T.step()
                        
                        R_rows_pivoted = np.asarray(qr_obj_A_T.R[:rank, :rank])  # k x k
                        selected_rows = np.asarray(qr_obj_A_T.chosen_columns[:rank])
                    
                    selected_rows = selected_rows[(selected_rows >= 0) & (selected_rows < N)]
                    
                    # Compute scipy QR of selected rows: A^T[:, selected_rows] = Q @ R
                    # This is equivalent to QR on rows of A
                    A_T_dense = A_dense.T
                    A_T_selected = A_T_dense[:, selected_rows]
                    Q_T_scipy, R_T_scipy = qr(A_T_selected, mode='economic')  # Q is (N, rank), R is (rank, rank)
                    
                    # Scale R_T_scipy rows so that r_ii > 0
                    for i in range(R_T_scipy.shape[0]):
                        if R_T_scipy[i, i] < 0:
                            R_T_scipy[i, :] *= -1
                            Q_T_scipy[:, i] *= -1
                    
                    # Compare R_rows
                    R_T_scipy_norm = np.linalg.norm(R_T_scipy, 'fro')
                    R_rows_pivoted_norm = np.linalg.norm(R_rows_pivoted, 'fro')
                    
                    if R_T_scipy_norm > 0:
                        r_rows_error = np.linalg.norm(R_rows_pivoted - R_T_scipy, 'fro') / R_T_scipy_norm
                    else:
                        r_rows_error = np.linalg.norm(R_rows_pivoted - R_T_scipy, 'fro')
                    
                    r_comparison_results['r_rows_errors'].append(float(r_rows_error))
                    
                    print(f"    R_rows relative error: {r_rows_error:.2e}")
                    
                    del cur_obj
                    if 'qr_obj_A_T' in locals():
                        del qr_obj_A_T
                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Plot R comparison errors
            if r_comparison_results['ranks']:
                print(f"\n{'='*70}")
                print("Creating R matrix comparison plots")
                print(f"{'='*70}")
                
                fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                
                ranks_plot = r_comparison_results['ranks']
                r_cols_errors = np.array(r_comparison_results['r_cols_errors'])
                r_rows_errors = np.array(r_comparison_results['r_rows_errors'])
                
                # Replace zeros with small value for log scale
                r_cols_errors = np.maximum(r_cols_errors, 1e-16)
                r_rows_errors = np.maximum(r_rows_errors, 1e-16)
                
                ax.semilogy(ranks_plot, r_cols_errors, 'o-', color='#9467bd', linewidth=2, markersize=8, label='R_cols (QR on columns)', alpha=0.8)
                ax.semilogy(ranks_plot, r_rows_errors, '^-', color='#8c564b', linewidth=2, markersize=8, label='R_rows (QR on rows via A^T)', alpha=0.8)
                
                # Set y-axis limits
                all_errors = np.concatenate([r_cols_errors, r_rows_errors])
                if len(all_errors) > 0:
                    y_min = np.percentile(all_errors, 1)
                    y_max = np.percentile(all_errors, 99)
                    y_min = max(1e-16, y_min * 0.3)
                    y_max = y_max * 3
                    ax.set_ylim(y_min, y_max)
                
                ax.set_xlabel('Rank', fontsize=12)
                ax.set_ylabel('Relative Frobenius Error', fontsize=12)
                ax.set_title('R Matrix Comparison: PivotedQR vs Scipy QR', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=11, loc='best')
                
                plt.tight_layout()
                plot_filename = f'{plot_dir}/r_matrix_comparison.png'
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"Saved: {plot_filename}")
                plt.close()
                
                # Save R comparison results to JSON
                r_results_dict = {
                    'experiment': experiment_name,
                    'dimension': dim,
                    'problem_type': problem_type,
                    'ranks': ranks_plot,
                    'r_cols_errors': [float(x) for x in r_comparison_results['r_cols_errors']],
                    'r_rows_errors': [float(x) for x in r_comparison_results['r_rows_errors']]
                }
                json_filename = f'{plot_dir}/r_comparison_results.json'
                with open(json_filename, 'w') as f:
                    json.dump(r_results_dict, f, indent=2)
                print(f"Saved: {json_filename}")
                
                # Also save as text file
                txt_filename = f'{plot_dir}/r_comparison_results.txt'
                with open(txt_filename, 'w') as f:
                    f.write(f"R Matrix Comparison Results\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Experiment: {experiment_name}\n")
                    f.write(f"Dimension: {dim}\n")
                    f.write(f"Problem Type: {problem_type}\n")
                    f.write(f"\n{'Rank':<8} {'R_cols Error':<20} {'R_rows Error':<20}\n")
                    f.write("-" * 80 + "\n")
                    for i, rank in enumerate(ranks_plot):
                        r_cols_err = r_comparison_results['r_cols_errors'][i]
                        r_rows_err = r_comparison_results['r_rows_errors'][i]
                        f.write(f"{rank:<8} {r_cols_err:<20.6e} {r_rows_err:<20.6e}\n")
                print(f"Saved: {txt_filename}")
    
    # Create plots
    print(f"\n{'='*70}")
    print("Creating plots")
    print(f"{'='*70}")
    
    # plot_dir and experiment_name already defined earlier in the function
    # (used for incremental saving and all outputs)
    
    # Define distinct colors, markers, and linestyles for each method
    colors = {'cur_incremental': '#2ca02c',          # green
              'randomized_svd': '#d62728',           # red
              'pivoted_qr_cur': '#9467bd',           # purple
              'iterative_cur_lu': '#17becf'}         # cyan
    
    # Define markers - use different markers for random vs greedy
    markers = {'cur_incremental_random': '^',
               'cur_incremental_greedy': 's',
               'cur_incremental': '^',  # fallback
               'randomized_svd': 'D',
               'pivoted_qr_cur_random': 'v',
               'pivoted_qr_cur_greedy': 'X',
               'pivoted_qr_cur': 'v',   # fallback
               'iterative_cur_lu': 'P'}
    
    linestyles = {'cur_incremental': '-.',
                  'randomized_svd': ':',
                  'pivoted_qr_cur': '-',
                  'iterative_cur_lu': '-.'}
    
    # Plot 1: Error vs Rank
    plt.figure(figsize=(14, 8))
    
    # Note: Each method's results include rank 0, so lines will naturally connect from ||A||
    
    # Plot CURIncremental for each sampling option
    for cur_sampling in cur_sampling_options:
        key = f'cur_incremental_{cur_sampling}'
        if key in results and results[key]['ranks']:
            ranks_plot = results[key]['ranks']
            errors_plot = results[key]['errors']
            if len(ranks_plot) > 0:
                label = f'CURIncremental ({cur_sampling})' if len(cur_sampling_options) > 1 else 'CURIncremental'
                plt.semilogy(ranks_plot, errors_plot, 
                            marker=markers.get(key, '^'),
                            linestyle=linestyles.get(key, '-'),
                            color=colors.get(key, '#2ca02c'),
                            label=label, 
                            linewidth=2.5, markersize=9, markevery=max(1, len(ranks_plot)//20))
    
    if results['randomized_svd']['ranks']:
        ranks_plot = results['randomized_svd']['ranks']
        errors_plot = results['randomized_svd']['errors']
        if len(ranks_plot) > 0:
            plt.semilogy(ranks_plot, errors_plot, 
                        marker=markers['randomized_svd'],
                        linestyle=linestyles['randomized_svd'],
                        color=colors['randomized_svd'],
                        label='randomized_svd', 
                        linewidth=2.5, markersize=9, markevery=max(1, len(ranks_plot)//20))
    
    # Plot pivoted_qr_cur for each sampling option
    for pqr_sampling in pivoted_qr_cur_sampling_options:
        key = f'pivoted_qr_cur_{pqr_sampling}'
        if key in results and results[key]['ranks']:
            ranks_plot = results[key]['ranks']
            # Use errors computed by estimate_approximation_error (2-norm error)
            errors_plot = results[key]['errors']
            if len(ranks_plot) > 0:
                label = f'pivoted_qr_cur ({pqr_sampling})' if len(pivoted_qr_cur_sampling_options) > 1 else 'pivoted_qr_cur'
                plt.semilogy(ranks_plot, errors_plot, 
                            marker=markers.get(key, 'v'),
                            linestyle=linestyles.get(key, '-'),
                            color=colors.get(key, '#9467bd'),
                            label=label, 
                            linewidth=2.5, markersize=9, markevery=max(1, len(ranks_plot)//20))

    # Plot IterativeCUR if present
    if 'iterative_cur_lu' in results and results['iterative_cur_lu']['ranks']:
        ranks_plot = results['iterative_cur_lu']['ranks']
        errors_plot = results['iterative_cur_lu']['errors']
        if len(ranks_plot) > 0:
            plt.semilogy(ranks_plot, errors_plot,
                        marker=markers['iterative_cur_lu'],
                        linestyle=linestyles['iterative_cur_lu'],
                        color=colors['iterative_cur_lu'],
                        label='iterative_cur_lu',
                        linewidth=2.5, markersize=9, markevery=max(1, len(ranks_plot)//20))
    
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Approximation Error', fontsize=12)
    plt.title(f'Approximation Error vs Rank ({problem_label})', fontsize=14)
    
    # Add vertical dashed lines for when all norms became negative
    for cur_sampling in cur_sampling_options:
        key = f'cur_incremental_{cur_sampling}'
        if key in results and results[key].get('iter_when_all_norms_are_negative', -1) >= 0:
            neg_iter = results[key]['iter_when_all_norms_are_negative']
            plt.axvline(x=neg_iter, color=colors.get(key, '#2ca02c'), linestyle='--', linewidth=2, alpha=0.7, 
                       label=f'CUR ({cur_sampling}): all norms negative at rank {neg_iter}')
    for pqr_sampling in pivoted_qr_cur_sampling_options:
        key = f'pivoted_qr_cur_{pqr_sampling}'
        if key in results and results[key].get('iter_when_all_norms_are_negative', -1) >= 0:
            neg_iter = results[key]['iter_when_all_norms_are_negative']
            plt.axvline(x=neg_iter, color=colors.get(key, '#9467bd'), linestyle='--', linewidth=2, alpha=0.7,
                       label=f'PivotedQR ({pqr_sampling}): all norms negative at rank {neg_iter}')
    
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_filename = f'{plot_dir}/error_vs_rank.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_filename}")
    plt.close()
    
    # Plot 2: Time vs Rank (with distinct colors and markers)
    plt.figure(figsize=(14, 8))
    # Plot CURIncremental for each sampling option
    for cur_sampling in cur_sampling_options:
        key = f'cur_incremental_{cur_sampling}'
        if key in results and results[key]['ranks']:
            ranks_plot = results[key]['ranks']
            times_plot = results[key]['times']
            rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0]
            if rank_nonzero_idx:
                label = f'CURIncremental ({cur_sampling})' if len(cur_sampling_options) > 1 else 'CURIncremental'
                plt.plot([ranks_plot[i] for i in rank_nonzero_idx], 
                        [times_plot[i] for i in rank_nonzero_idx], 
                        marker=markers.get(key, '^'),
                        linestyle=linestyles.get(key, '-'),
                        color=colors.get(key, '#2ca02c'),
                        label=label, 
                        linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
    if results['randomized_svd']['ranks']:
        ranks_plot = results['randomized_svd']['ranks']
        times_plot = results['randomized_svd']['times']
        rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0]
        if rank_nonzero_idx:
            plt.plot([ranks_plot[i] for i in rank_nonzero_idx], 
                    [times_plot[i] for i in rank_nonzero_idx], 
                    marker=markers['randomized_svd'],
                    linestyle=linestyles['randomized_svd'],
                    color=colors['randomized_svd'],
                    label='randomized_svd', 
                    linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
    
    # Plot pivoted_qr_cur for each sampling option
    for pqr_sampling in pivoted_qr_cur_sampling_options:
        key = f'pivoted_qr_cur_{pqr_sampling}'
        if key in results and results[key]['ranks']:
            ranks_plot = results[key]['ranks']
            times_plot = results[key]['times']
            rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0]
            if rank_nonzero_idx:
                label = f'pivoted_qr_cur ({pqr_sampling})' if len(pivoted_qr_cur_sampling_options) > 1 else 'pivoted_qr_cur'
                plt.plot([ranks_plot[i] for i in rank_nonzero_idx], 
                        [times_plot[i] for i in rank_nonzero_idx], 
                        marker=markers.get(key, 'v'),
                        linestyle=linestyles.get(key, '-'),
                        color=colors.get(key, '#9467bd'),
                        label=label, 
                        linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))

    if 'iterative_cur_lu' in results and results['iterative_cur_lu']['ranks']:
        ranks_plot = results['iterative_cur_lu']['ranks']
        times_plot = results['iterative_cur_lu']['times']
        rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0]
        if rank_nonzero_idx:
            plt.plot([ranks_plot[i] for i in rank_nonzero_idx],
                    [times_plot[i] for i in rank_nonzero_idx],
                    marker=markers['iterative_cur_lu'],
                    linestyle=linestyles['iterative_cur_lu'],
                    color=colors['iterative_cur_lu'],
                    label='iterative_cur_lu',
                    linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
    
    plt.xlabel('Rank', fontsize=14)
    plt.ylabel('Build Time (seconds)', fontsize=14)
    plt.title(f'Build Time vs Rank ({problem_label})', fontsize=16, fontweight='bold')
    
    # Add vertical dashed lines for when all norms became negative
    for cur_sampling in cur_sampling_options:
        key = f'cur_incremental_{cur_sampling}'
        if key in results and results[key].get('iter_when_all_norms_are_negative', -1) >= 0:
            neg_iter = results[key]['iter_when_all_norms_are_negative']
            plt.axvline(x=neg_iter, color=colors.get(key, '#2ca02c'), linestyle='--', linewidth=2, alpha=0.7, 
                       label=f'CUR ({cur_sampling}): all norms negative at rank {neg_iter}')
    for pqr_sampling in pivoted_qr_cur_sampling_options:
        key = f'pivoted_qr_cur_{pqr_sampling}'
        if key in results and results[key].get('iter_when_all_norms_are_negative', -1) >= 0:
            neg_iter = results[key]['iter_when_all_norms_are_negative']
            plt.axvline(x=neg_iter, color=colors.get(key, '#9467bd'), linestyle='--', linewidth=2, alpha=0.7,
                       label=f'PivotedQR ({pqr_sampling}): all norms negative at rank {neg_iter}')
    
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_filename = f'{plot_dir}/time_vs_rank.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_filename}")
    plt.close()
    
    # Plot 3: Error vs Time (efficiency plot)
    plt.figure(figsize=(14, 8))
    # Plot CURIncremental for each sampling option
    for cur_sampling in cur_sampling_options:
        key = f'cur_incremental_{cur_sampling}'
        if key in results and results[key]['ranks']:
            ranks_plot = results[key]['ranks']
            times_plot = results[key]['times']
            errors_plot = results[key]['errors']
            rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0]
            if rank_nonzero_idx:
                label = f'CURIncremental ({cur_sampling})' if len(cur_sampling_options) > 1 else 'CURIncremental'
                plt.loglog([times_plot[i] for i in rank_nonzero_idx], 
                          [errors_plot[i] for i in rank_nonzero_idx], 
                          marker=markers.get(key, '^'),
                          linestyle=linestyles.get(key, '-'),
                          color=colors.get(key, '#2ca02c'),
                          label=label, 
                          linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
    if results['randomized_svd']['ranks']:
        ranks_plot = results['randomized_svd']['ranks']
        times_plot = results['randomized_svd']['times']
        errors_plot = results['randomized_svd']['errors']
        rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0]
        if rank_nonzero_idx:
            plt.loglog([times_plot[i] for i in rank_nonzero_idx], 
                      [errors_plot[i] for i in rank_nonzero_idx], 
                      marker=markers['randomized_svd'],
                      linestyle=linestyles['randomized_svd'],
                      color=colors['randomized_svd'],
                      label='randomized_svd', 
                      linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
    
    # Plot pivoted_qr_cur for each sampling option
    for pqr_sampling in pivoted_qr_cur_sampling_options:
        key = f'pivoted_qr_cur_{pqr_sampling}'
        if key in results and results[key]['ranks']:
            ranks_plot = results[key]['ranks']
            times_plot = results[key]['times']
            errors_plot = results[key]['errors']
            rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0]
            if rank_nonzero_idx:
                label = f'pivoted_qr_cur ({pqr_sampling})' if len(pivoted_qr_cur_sampling_options) > 1 else 'pivoted_qr_cur'
                plt.loglog([times_plot[i] for i in rank_nonzero_idx], 
                          [errors_plot[i] for i in rank_nonzero_idx], 
                          marker=markers.get(key, 'v'),
                          linestyle=linestyles.get(key, '-'),
                          color=colors.get(key, '#9467bd'),
                          label=label, 
                          linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
    
    plt.xlabel('Build Time (seconds)', fontsize=14)
    plt.ylabel('Approximation Error', fontsize=14)
    plt.title(f'Error vs Time ({problem_label})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_filename = f'{plot_dir}/error_vs_time.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_filename}")
    plt.close()
    
    # Plot 4: Frobenius Error vs Rank (if Frobenius errors are available)
    has_frobenius_data = any(
        results.get(key, {}).get('errors_frobenius') is not None 
        and any(err is not None for err in results[key].get('errors_frobenius', []))
        for key in results.keys()
    )
    
    if has_frobenius_data:
        plt.figure(figsize=(14, 8))
        
        # Plot rank 0 separately with different marker (shared by all methods)
        first_key = list(results.keys())[0]
        if results[first_key].get('errors_frobenius') and results[first_key]['errors_frobenius'][0] is not None:
            rank_0_error_fro = results[first_key]['errors_frobenius'][0]
            plt.semilogy([0], [rank_0_error_fro], 'ko', label='rank 0 (||A||_F)', 
                        linewidth=2, markersize=12, markerfacecolor='none', 
                        markeredgewidth=2.5, alpha=0.8, zorder=10)
        
        # Plot CURIncremental for each sampling option
        for cur_sampling in cur_sampling_options:
            key = f'cur_incremental_{cur_sampling}'
            if key in results and results[key]['ranks']:
                ranks_plot = results[key]['ranks']
                errors_fro_plot = results[key].get('errors_frobenius')
                if errors_fro_plot is None:
                    errors_fro_plot = [None] * len(ranks_plot)
                rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0 and i < len(errors_fro_plot) and errors_fro_plot[i] is not None]
                if rank_nonzero_idx:
                    label = f'CURIncremental ({cur_sampling})' if len(cur_sampling_options) > 1 else 'CURIncremental'
                    plt.semilogy([ranks_plot[i] for i in rank_nonzero_idx], 
                                [errors_fro_plot[i] for i in rank_nonzero_idx], 
                                marker=markers.get(key, '^'),
                                linestyle=linestyles.get(key, '-'),
                                color=colors.get(key, '#2ca02c'),
                                label=label, 
                                linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
        
        if results['randomized_svd']['ranks']:
            ranks_plot = results['randomized_svd']['ranks']
            errors_fro_plot = results['randomized_svd'].get('errors_frobenius')
            if errors_fro_plot is None:
                errors_fro_plot = [None] * len(ranks_plot)
            rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0 and i < len(errors_fro_plot) and errors_fro_plot[i] is not None]
            if rank_nonzero_idx:
                plt.semilogy([ranks_plot[i] for i in rank_nonzero_idx], 
                            [errors_fro_plot[i] for i in rank_nonzero_idx], 
                            marker=markers['randomized_svd'],
                            linestyle=linestyles['randomized_svd'],
                            color=colors['randomized_svd'],
                            label='randomized_svd', 
                            linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
        
        # Plot pivoted_qr_cur for each sampling option
        for pqr_sampling in pivoted_qr_cur_sampling_options:
            key = f'pivoted_qr_cur_{pqr_sampling}'
            if key in results and results[key]['ranks']:
                ranks_plot = results[key]['ranks']
                errors_fro_plot = results[key].get('errors_frobenius')
                if errors_fro_plot is None:
                    errors_fro_plot = [None] * len(ranks_plot)
                rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0 and i < len(errors_fro_plot) and errors_fro_plot[i] is not None]
                if rank_nonzero_idx:
                    label = f'pivoted_qr_cur ({pqr_sampling})' if len(pivoted_qr_cur_sampling_options) > 1 else 'pivoted_qr_cur'
                    plt.semilogy([ranks_plot[i] for i in rank_nonzero_idx], 
                                [errors_fro_plot[i] for i in rank_nonzero_idx], 
                                marker=markers.get(key, 'v'),
                                linestyle=linestyles.get(key, '-'),
                                color=colors.get(key, '#9467bd'),
                                label=label, 
                                linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
        
        plt.xlabel('Rank', fontsize=12)
        plt.ylabel('Frobenius Norm Error', fontsize=12)
        plt.title(f'Frobenius Norm Error vs Rank ({problem_label})', fontsize=14)
        
        # Add vertical dashed lines for when all norms became negative
        for cur_sampling in cur_sampling_options:
            key = f'cur_incremental_{cur_sampling}'
            if key in results and results[key].get('iter_when_all_norms_are_negative', -1) >= 0:
                neg_iter = results[key]['iter_when_all_norms_are_negative']
                plt.axvline(x=neg_iter, color=colors.get(key, '#2ca02c'), linestyle='--', linewidth=2, alpha=0.7, 
                           label=f'CUR ({cur_sampling}): all norms negative at rank {neg_iter}')
        for pqr_sampling in pivoted_qr_cur_sampling_options:
            key = f'pivoted_qr_cur_{pqr_sampling}'
            if key in results and results[key].get('iter_when_all_norms_are_negative', -1) >= 0:
                neg_iter = results[key]['iter_when_all_norms_are_negative']
                plt.axvline(x=neg_iter, color=colors.get(key, '#9467bd'), linestyle='--', linewidth=2, alpha=0.7,
                           label=f'PivotedQR ({pqr_sampling}): all norms negative at rank {neg_iter}')
        
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_filename = f'{plot_dir}/frobenius_error_vs_rank.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_filename}")
        plt.close()
        
        # Plot 5: Comparison of 2-norm vs Frobenius norm errors (if both available)
        plt.figure(figsize=(14, 8))
        
        # Plot rank 0 separately
        first_key = list(results.keys())[0]
        rank_0_error = results[first_key]['errors'][0]
        if results[first_key].get('errors_frobenius') and results[first_key]['errors_frobenius'][0] is not None:
            rank_0_error_fro = results[first_key]['errors_frobenius'][0]
            plt.semilogy([0], [rank_0_error], 'ko', label='rank 0 (2-norm)', 
                        linewidth=2, markersize=12, markerfacecolor='none', 
                        markeredgewidth=2.5, alpha=0.8, zorder=10)
            plt.semilogy([0], [rank_0_error_fro], 'ks', label='rank 0 (Frobenius)', 
                        linewidth=2, markersize=12, markerfacecolor='none', 
                        markeredgewidth=2.5, alpha=0.8, zorder=10)
        else:
            plt.semilogy([0], [rank_0_error], 'ko', label='rank 0 (||A||)', 
                        linewidth=2, markersize=12, markerfacecolor='none', 
                        markeredgewidth=2.5, alpha=0.8, zorder=10)
        
        # Plot both 2-norm and Frobenius for each method
        for cur_sampling in cur_sampling_options:
            key = f'cur_incremental_{cur_sampling}'
            if key in results and results[key]['ranks']:
                ranks_plot = results[key]['ranks']
                errors_plot = results[key]['errors']
                errors_fro_plot = results[key].get('errors_frobenius')
                if errors_fro_plot is None:
                    errors_fro_plot = [None] * len(ranks_plot)
                rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0]
                if rank_nonzero_idx:
                    label = f'CURIncremental ({cur_sampling})' if len(cur_sampling_options) > 1 else 'CURIncremental'
                    # 2-norm error
                    plt.semilogy([ranks_plot[i] for i in rank_nonzero_idx], 
                                [errors_plot[i] for i in rank_nonzero_idx], 
                                marker=markers.get(key, '^'),
                                linestyle=linestyles.get(key, '-'),
                                color=colors.get(key, '#2ca02c'),
                                label=f'{label} (2-norm)', 
                                linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
                    # Frobenius error (if available)
                    fro_nonzero_idx = [i for i in rank_nonzero_idx if i < len(errors_fro_plot) and errors_fro_plot[i] is not None]
                    if fro_nonzero_idx:
                        plt.semilogy([ranks_plot[i] for i in fro_nonzero_idx], 
                                    [errors_fro_plot[i] for i in fro_nonzero_idx], 
                                    marker=markers.get(key, '^'),
                                    linestyle='--',
                                    color=colors.get(key, '#2ca02c'),
                                    alpha=0.7,
                                    label=f'{label} (Frobenius)', 
                                    linewidth=2.5, markersize=9, markevery=max(1, len(fro_nonzero_idx)//20))
        
        for pqr_sampling in pivoted_qr_cur_sampling_options:
            key = f'pivoted_qr_cur_{pqr_sampling}'
            if key in results and results[key]['ranks']:
                ranks_plot = results[key]['ranks']
                errors_plot = results[key]['errors']
                errors_fro_plot = results[key].get('errors_frobenius')
                if errors_fro_plot is None:
                    errors_fro_plot = [None] * len(ranks_plot)
                rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0]
                if rank_nonzero_idx:
                    label = f'pivoted_qr_cur ({pqr_sampling})' if len(pivoted_qr_cur_sampling_options) > 1 else 'pivoted_qr_cur'
                    # 2-norm error
                    plt.semilogy([ranks_plot[i] for i in rank_nonzero_idx], 
                                [errors_plot[i] for i in rank_nonzero_idx], 
                                marker=markers.get(key, 'v'),
                                linestyle=linestyles.get(key, '-'),
                                color=colors.get(key, '#9467bd'),
                                label=f'{label} (2-norm)', 
                                linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
                    # Frobenius error (if available)
                    fro_nonzero_idx = [i for i in rank_nonzero_idx if i < len(errors_fro_plot) and errors_fro_plot[i] is not None]
                    if fro_nonzero_idx:
                        plt.semilogy([ranks_plot[i] for i in fro_nonzero_idx], 
                                    [errors_fro_plot[i] for i in fro_nonzero_idx], 
                                    marker=markers.get(key, 'v'),
                                    linestyle='--',
                                    color=colors.get(key, '#9467bd'),
                                    alpha=0.7,
                                    label=f'{label} (Frobenius)', 
                                    linewidth=2.5, markersize=9, markevery=max(1, len(fro_nonzero_idx)//20))
        
        if results['randomized_svd']['ranks']:
            ranks_plot = results['randomized_svd']['ranks']
            errors_plot = results['randomized_svd']['errors']
            errors_fro_plot = results['randomized_svd'].get('errors_frobenius')
            if errors_fro_plot is None:
                errors_fro_plot = [None] * len(ranks_plot)
            rank_nonzero_idx = [i for i, r in enumerate(ranks_plot) if r > 0]
            if rank_nonzero_idx:
                # 2-norm error
                plt.semilogy([ranks_plot[i] for i in rank_nonzero_idx], 
                            [errors_plot[i] for i in rank_nonzero_idx], 
                            marker=markers['randomized_svd'],
                            linestyle=linestyles['randomized_svd'],
                            color=colors['randomized_svd'],
                            label='randomized_svd (2-norm)', 
                            linewidth=2.5, markersize=9, markevery=max(1, len(rank_nonzero_idx)//20))
                # Frobenius error (if available)
                fro_nonzero_idx = [i for i in rank_nonzero_idx if i < len(errors_fro_plot) and errors_fro_plot[i] is not None]
                if fro_nonzero_idx:
                    plt.semilogy([ranks_plot[i] for i in fro_nonzero_idx], 
                                [errors_fro_plot[i] for i in fro_nonzero_idx], 
                                marker=markers['randomized_svd'],
                                linestyle='--',
                                color=colors['randomized_svd'],
                                alpha=0.7,
                                label='randomized_svd (Frobenius)', 
                                linewidth=2.5, markersize=9, markevery=max(1, len(fro_nonzero_idx)//20))
        
        plt.xlabel('Rank', fontsize=12)
        plt.ylabel('Approximation Error', fontsize=12)
        plt.title(f'2-Norm vs Frobenius Norm Error Comparison ({problem_label})', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_filename = f'{plot_dir}/error_2norm_vs_frobenius_comparison.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_filename}")
        plt.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Rank':<8} {'Error':<15} {'Time (s)':<12}")
    print("-" * 70)
    
    for method_name, method_results in results.items():
        for i, rank in enumerate(method_results['ranks']):
            error = method_results['errors'][i]
            time_val = method_results['times'][i]
            print(f"{method_name:<25} {rank:<8} {error:<15.2e} {time_val:<12.3f}")
    
    # Final save
    if output_file:
        save_comparison_results(results, metadata, output_file, verbose=True)
    
    return results, metadata, output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare CURIncremental vs PivotedQRCUR vs randomized_svd')
    parser.add_argument('--dim', type=int, default=128, help='Problem size (dim x dim x dim) for Toeplitz')
    parser.add_argument('--ranks', type=str, nargs='+', default=None, 
                       help='Ranks to test (default: [10, 20, 30, 40, 50]). Can also use numpy-style ranges like 0:1024:32')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output file name for results JSON (optional)')
    parser.add_argument('--problem-type', type=str, default='toeplitz', choices=['toeplitz', 'graph', 'line-graph', 'dense'],
                       help='Problem type: toeplitz (3D convolution), graph (sparse matrix), line-graph (path graph adjacency), or dense (dense matrix with slow decay)')
    parser.add_argument(
        '--graph-name',
        type=str,
        default='FullChip',
        choices=[
            'FullChip',
            'circuit5M_dc',
            'mawi_201512012345',
            'mawi_201512020000',
            'StocF-1465',
            'Freescale2',
            'Emilia_923',
            'Long_Coup_dt0',
            'wiki-Talk',
            'rajat31',
        ],
        help='Graph name if problem-type=graph. Restricted to the 10 SuiteSparse matrices listed in manuscript/paper.tex (Table tab:suitesparse-links).',
    )
    parser.add_argument('--graphs-dir', type=str, default=None,
                       help='Directory where graphs are stored (if None, uses current directory)')
    parser.add_argument('--gpu-id', type=int, default=2,
                       help='GPU ID to use (sets CUDA_VISIBLE_DEVICES). Default: 2')
    parser.add_argument('--test-norm-accuracy', action='store_true', default=False,
                       help='Run norm accuracy and R matrix comparison tests (only for dense matrices). Default: False')
    parser.add_argument('--t-method', type=str, default='svd', choices=['svd', 'pinv', 'direct'],
                       help='Method for computing T matrix in PivotedQRCUR: svd (use_svd_for_T=True), pinv (compute_T_via_pinv=True), or direct (both False). Default: svd')
    parser.add_argument('--skip-svd', action='store_true', default=False,
                       help='Skip randomized_svd test entirely. Default: False')
    parser.add_argument('--skip-cur', action='store_true', default=False,
                       help='Skip CURIncremental test entirely. Default: False')
    parser.add_argument('--skip-pivoted-qr', action='store_true', default=False,
                       help='Skip pivoted_qr_cur test entirely. Default: False')
    parser.add_argument('--only-cur', action='store_true', default=False,
                       help='Run ONLY CURIncremental (skip SVD and pivoted_qr_cur). Default: False')
    parser.add_argument('--only-pivoted-qr', action='store_true', default=False,
                       help='Run ONLY pivoted_qr_cur (skip SVD and CURIncremental). Default: False')
    parser.add_argument('--cur-sampling', type=str, nargs='+', default=['random'],
                       choices=['random', 'greedy', 'uniform'],
                       help='Sampling strategies for CURIncremental. Can specify multiple: --cur-sampling random greedy. Default: random')
    parser.add_argument('--pivoted-qr-cur-sampling', type=str, nargs='+', default=['random'],
                       choices=['random', 'greedy', 'uniform'],
                       help='Sampling strategies for pivoted_qr_cur. Can specify multiple: --pivoted-qr-cur-sampling random greedy. Default: random')
    parser.add_argument('--compute-frobenius', action='store_true', default=False,
                       help='Force computation of Frobenius norm error (uses memory-efficient row-by-row method). Default: False')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for random sampling methods. Default: 42')
    parser.add_argument('--svd-oversample', type=int, default=5,
                       help='Number of oversamples for randomized SVD. Default: 5')
    parser.add_argument('--svd-power-iter', type=int, default=0,
                       help='Number of power iterations for randomized SVD. Default: 0')
    parser.add_argument('--svd-batch-matvec', type=lambda x: x.lower() == 'true', default=None,
                       help='Use batch matvec for randomized SVD. If not specified, uses True for graph problems, False for toeplitz. Default: None (auto-detect)')
    
    args = parser.parse_args()
    
    # Set GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Using GPU: {args.gpu_id}")
    
    if args.ranks is None:
        args.ranks = list(np.arange(0, 1024, 100))
    elif len(args.ranks) == 1 and ':' in str(args.ranks[0]):
        # Support numpy-style range: start:stop:step
        range_str = str(args.ranks[0])
        parts = range_str.split(':')
        if len(parts) == 3:
            start, stop, step = map(int, parts)
            args.ranks = list(range(start, stop + 1, step))
        elif len(parts) == 2:
            start, stop = map(int, parts)
            args.ranks = list(range(start, stop + 1))
    else:
        # Convert list of strings to integers
        args.ranks = [int(r) for r in args.ranks]
    
    # Handle --only-* flags by setting skip flags
    skip_svd = args.skip_svd or args.only_cur or args.only_pivoted_qr
    skip_cur = args.skip_cur or args.only_pivoted_qr
    skip_pivoted_qr = args.skip_pivoted_qr or args.only_cur
    
    # Set batch_matvec based on problem type if not explicitly provided
    if args.svd_batch_matvec is None:
        # Use batch_matvec for graph problems, not for toeplitz
        svd_batch_matvec = (args.problem_type in ['graph', 'line-graph'])
    else:
        svd_batch_matvec = args.svd_batch_matvec
    
    results, metadata, output_file = test_comparison(
        dim=args.dim, 
        ranks=args.ranks, 
        problem_type=args.problem_type,
        graph_name=args.graph_name,
        output_file=args.output,
        test_norm_accuracy=args.test_norm_accuracy,
        t_method=args.t_method,
        skip_svd=skip_svd,
        skip_cur=skip_cur,
        skip_pivoted_qr=skip_pivoted_qr,
        cur_sampling_options=args.cur_sampling,
        pivoted_qr_cur_sampling_options=args.pivoted_qr_cur_sampling,
        graphs_dir=args.graphs_dir,
        force_frobenius=args.compute_frobenius,
        random_seed=args.seed,
        svd_oversample=args.svd_oversample,
        svd_power_iter=args.svd_power_iter,
        svd_batch_matvec=svd_batch_matvec
    )
    
    if output_file:
        print(f"\nResults saved to: {output_file}")
    
    print("\nDone!")
