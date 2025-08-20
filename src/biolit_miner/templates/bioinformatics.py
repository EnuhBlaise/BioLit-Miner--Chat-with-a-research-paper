# Bioinformatics Analysis Template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO, Seq, Align
from Bio.SeqUtils import GC, molecular_weight
from Bio.Blast import NCBIWWW, NCBIXML
import requests
import json

def load_sequence_data(file_path, format='fasta'):
    """Load sequence data from various formats"""
    sequences = []
    for record in SeqIO.parse(file_path, format):
        sequences.append({
            'id': record.id,
            'description': record.description,
            'sequence': str(record.seq),
            'length': len(record.seq)
        })
    return pd.DataFrame(sequences)

def sequence_statistics(sequences_df):
    """Calculate basic sequence statistics"""
    stats = {}
    
    for idx, row in sequences_df.iterrows():
        seq = Seq.Seq(row['sequence'])
        stats[row['id']] = {
            'length': len(seq),
            'gc_content': GC(seq),
            'molecular_weight': molecular_weight(seq, 'DNA'),
            'nucleotide_counts': {
                'A': seq.count('A'),
                'T': seq.count('T'),
                'G': seq.count('G'),
                'C': seq.count('C')
            }
        }
    
    stats_df = pd.DataFrame(stats).T
    print("Sequence Statistics:")
    print(stats_df)
    return stats_df

def gc_content_analysis(sequences_df):
    """Analyze GC content distribution"""
    gc_contents = []
    
    for idx, row in sequences_df.iterrows():
        gc = GC(row['sequence'])
        gc_contents.append(gc)
    
    plt.figure(figsize=(10, 6))
    plt.hist(gc_contents, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('GC Content (%)')
    plt.ylabel('Frequency')
    plt.title('GC Content Distribution')
    plt.axvline(np.mean(gc_contents), color='red', linestyle='--', label=f'Mean: {np.mean(gc_contents):.2f}%')
    plt.legend()
    plt.show()
    
    return gc_contents

def sequence_alignment(seq1, seq2, match_score=2, mismatch_score=-1, gap_score=-1):
    """Perform pairwise sequence alignment"""
    aligner = Align.PairwiseAligner()
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = gap_score
    aligner.extend_gap_score = gap_score
    
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]
    
    print("Best Alignment:")
    print(best_alignment)
    print(f"Alignment Score: {best_alignment.score}")
    
    return best_alignment

def blast_search(sequence, database='nt', max_hits=10):
    """Perform BLAST search (requires internet connection)"""
    try:
        result_handle = NCBIWWW.qblast("blastn", database, sequence)
        blast_records = NCBIXML.parse(result_handle)
        
        hits = []
        for blast_record in blast_records:
            for alignment in blast_record.alignments[:max_hits]:
                for hsp in alignment.hsps:
                    hits.append({
                        'title': alignment.title,
                        'length': alignment.length,
                        'e_value': hsp.expect,
                        'score': hsp.score,
                        'identities': hsp.identities,
                        'query_start': hsp.query_start,
                        'query_end': hsp.query_end
                    })
        
        return pd.DataFrame(hits)
    
    except Exception as e:
        print(f"BLAST search failed: {e}")
        return pd.DataFrame()

def motif_analysis(sequences_df, motif_pattern):
    """Search for motifs in sequences"""
    import re
    
    motif_results = []
    
    for idx, row in sequences_df.iterrows():
        sequence = row['sequence']
        matches = list(re.finditer(motif_pattern, sequence, re.IGNORECASE))
        
        for match in matches:
            motif_results.append({
                'sequence_id': row['id'],
                'motif': match.group(),
                'start_position': match.start(),
                'end_position': match.end()
            })
    
    motif_df = pd.DataFrame(motif_results)
    print(f"Found {len(motif_df)} motif matches")
    return motif_df

def reading_frame_analysis(sequence):
    """Analyze all six reading frames"""
    frames = {}
    
    # Forward frames
    for i in range(3):
        frame_seq = sequence[i:]
        # Remove incomplete codons
        frame_seq = frame_seq[:len(frame_seq)//3*3]
        frames[f'Frame +{i+1}'] = frame_seq
    
    # Reverse complement frames
    rev_comp = str(Seq.Seq(sequence).reverse_complement())
    for i in range(3):
        frame_seq = rev_comp[i:]
        frame_seq = frame_seq[:len(frame_seq)//3*3]
        frames[f'Frame -{i+1}'] = frame_seq
    
    return frames

def codon_usage_analysis(sequences_df):
    """Analyze codon usage patterns"""
    from collections import Counter
    
    all_codons = []
    
    for idx, row in sequences_df.iterrows():
        sequence = row['sequence']
        # Extract codons (assuming sequence starts at reading frame)
        codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
        all_codons.extend(codons)
    
    codon_counts = Counter(all_codons)
    codon_df = pd.DataFrame(list(codon_counts.items()), columns=['Codon', 'Count'])
    codon_df = codon_df.sort_values('Count', ascending=False)
    
    plt.figure(figsize=(15, 8))
    top_codons = codon_df.head(20)
    plt.bar(top_codons['Codon'], top_codons['Count'])
    plt.xlabel('Codon')
    plt.ylabel('Frequency')
    plt.title('Top 20 Most Frequent Codons')
    plt.xticks(rotation=45)
    plt.show()
    
    return codon_df

def translate_sequences(sequences_df):
    """Translate DNA sequences to proteins"""
    translated = []
    
    for idx, row in sequences_df.iterrows():
        dna_seq = Seq.Seq(row['sequence'])
        protein_seq = dna_seq.translate()
        
        translated.append({
            'original_id': row['id'],
            'protein_id': f"{row['id']}_protein",
            'protein_sequence': str(protein_seq),
            'protein_length': len(protein_seq)
        })
    
    return pd.DataFrame(translated)

# Template usage example
if __name__ == "__main__":
    # Load sequence data
    # sequences_df = load_sequence_data("sequences.fasta")
    
    # Basic statistics
    # stats = sequence_statistics(sequences_df)
    
    # GC content analysis
    # gc_contents = gc_content_analysis(sequences_df)
    
    # Sequence alignment
    # seq1 = "ATGCGATCGATCG"
    # seq2 = "ATGCGATCGATCG"
    # alignment = sequence_alignment(seq1, seq2)
    
    # Motif analysis
    # motifs = motif_analysis(sequences_df, "ATG")  # Start codon
    
    # Codon usage
    # codon_usage = codon_usage_analysis(sequences_df)
    
    # Translation
    # proteins = translate_sequences(sequences_df)
    
    print("Bioinformatics analysis template ready for customization")