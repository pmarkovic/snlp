from Bio import SeqIO
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import Bio


def sample_records(genome_loc: Path, genome_red_loc: Path, num_records: int):
    """ Samples n reads from a fasta file and saves them to a new file.

    :param genome_loc: path to the unreduced file
    :param genome_red_loc: path to the reduced file
    :num_records: number of reads to sample
    """
    sequence = SeqIO.parse(genome_loc.absolute(), 'fasta')
    sequence = list(sequence)

    genome_red_loc.touch()
    handle = genome_red_loc.open('w')
    
    for _ in range(num_records):
        index = np.random.randint(len(sequence))
        SeqIO.write(sequence[index], handle, 'fasta')
    

def get_k_mers(genome_red_loc: Path, k: int) -> List[str]:
    """ Samples k-mers from a fasta file (preferrably the reduced one).
        See also https://en.wikipedia.org/wiki/K-mer
    :param genome_loc: path to the fasta file
    :param k: length of of the n-mer
    :return: a list of n-mers
    """
    sequence = SeqIO.parse(genome_red_loc.absolute(), 'fasta')
    sequence = list(sequence)

    final_sequence = []
    for record in sequence:
       final_sequence.append(str(record.seq).upper())

    final_sequence = "".join(final_sequence) # combine all these sequences into a continuous string of nucleotides
    kmer = []

    for i in range(len(final_sequence)-k+1):
        kmer.append(final_sequence[i:i+k])

    return kmer


def get_k_mers_24(genome_red_loc: Path, k: int, tandem_repeats=False) -> List[str]:
    """ Samples k-mers from a fasta file (preferrably the reduced one), but this time 
        only for tandem repeat regions or non tandem repeat regions.

    :param genome_loc: path to the fasta file
    :param k: length of of the n-mer
    :param tandem_repeats: get only tandem repeats or non-tandem repeats
    :return: a list of n-mers
    """


def k_mer_statistics(genome_red_loc: Path, K: int, delta=1.e-10) -> Tuple:
    """ Calculates relative k-mer frequencies and conditional k-mer probabilities 
        on the provided fasta file.

    :param genome_red_loc: path to the fasta file
    :param K: upper bound of the k of k-mers
    :param delta: threshold for probability mass loss, defaults to 1.e-10
    :return: lists of relative frequencies and conditional probabilities
    """
    rel_freq = defaultdict(dict)
    for k in range(K):
        kmer = get_k_mers(genome_red_loc, k)
        kmer_len = len(kmer)
        rel_freq[k] = Counter(kmer)
        for key in rel_freq[k]:
            rel_freq[k][key] /=kmer_len
        break
    print(rel_freq)


def k_mer_statistics_24(genome_red_loc: Path, K: int, tandem_repeats=False, delta=1.e-10) -> Tuple:
    """ Calculates relative k-mer frequencies and conditional k-mer probabilities 
        on the provided fasta file, but this time only for tandem repeat regions 
        or non tandem repeat regions.

    :param genome_red_loc: path to the fasta file
    :param K: upper bound of the k of k-mers
    :param tandem_repeats: get only tandem repeats or non-tandem repeats
    :param delta: threshold for probability mass loss, defaults to 1.e-10
    :return: lists of relative frequencies and conditional probabilities
    """


def conditional_entropy(rel_freqs: Dict, cond_probs: Dict) -> float:
    """ Calculates the conditional entropy of a corpus given by relative k-mer frequencies
        and conditional k-mer probabilities

    :param rel_freqs: (a dictionary of) relative frequencies
    :param cond_probs: (a dictionary of) conditional probabilities
    :return: the conditional entropy of the corpus
    """


def plot_k_mers(rel_freqs: List[Dict], n=10, k=5):
    """ Plots n most frequent k-mers vs. their frequency.

    :param rel_freqs: the list of relative frequency dicts
    :param n: the number of most frequent k-mers to plot
    :param k: the k of k-mers
    """


def plot_conditional_entropies(H_ks:List[float]):
    """ Plots conditional entropy vs. k-mer length

    :param H_ks: the conditional entropy scores
    """
if __name__ == '__main__':
    genome_loc = Path("./data/genome.fa")
    genome_red_loc = Path("./data/genome_reduced.fa")
    sample_records(genome_loc, genome_red_loc, 100)
    print(get_k_mers(genome_red_loc, 2))