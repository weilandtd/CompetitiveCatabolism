"""
Utility functions for amino acid analysis.
"""

from collections import Counter
from pathlib import Path
import pandas as pd


def analyze_amino_acid_frequency(fasta_file):
    """
    Analyze amino acid frequency from a FASTA file.
    
    Parameters:
    -----------
    fasta_file : str or Path
        Path to the FASTA file
    
    Returns:
    --------
    tuple: (amino_acid_counts, total_proteins, total_aa)
        - amino_acid_counts: Counter object with amino acid frequencies
        - total_proteins: number of protein sequences analyzed
        - total_aa: total number of amino acids counted
    """
    
    aa_counter = Counter()
    protein_count = 0
    current_sequence = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # New sequence header - process previous sequence
                if current_sequence:
                    seq = ''.join(current_sequence)
                    aa_counter.update(seq.upper())
                    protein_count += 1
                current_sequence = []
            else:
                # Sequence line
                current_sequence.append(line)
        
        # Process last sequence
        if current_sequence:
            seq = ''.join(current_sequence)
            aa_counter.update(seq.upper())
            protein_count += 1
    
    total_aa = sum(aa_counter.values())
    
    return aa_counter, protein_count, total_aa


def get_amino_acid_names():
    """
    Get a dictionary mapping single-letter amino acid codes to full names.
    
    Returns:
    --------
    dict: Mapping of single-letter code to full amino acid name (lowercase)
    """
    return {
        'A': 'alanine', 'R': 'arginine', 'N': 'asparagine', 'D': 'aspartate',
        'C': 'cysteine', 'Q': 'glutamine', 'E': 'glutamate', 'G': 'glycine',
        'H': 'histidine', 'I': 'isoleucine', 'L': 'leucine', 'K': 'lysine',
        'M': 'methionine', 'F': 'phenylalanine', 'P': 'proline', 'S': 'serine',
        'T': 'threonine', 'W': 'tryptophan', 'Y': 'tyrosine', 'V': 'valine'
    }


def get_amino_acid_to_code():
    """
    Get a dictionary mapping amino acid names (lowercase) to single-letter codes.
    
    Returns:
    --------
    dict: Mapping of amino acid name to single-letter code
    """
    names = get_amino_acid_names()
    return {v: k for k, v in names.items()}
