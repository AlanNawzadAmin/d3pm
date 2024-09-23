import os
import gzip
import json
import random
import numpy as np
import requests
import tarfile
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)

def extract_uniref50(tar_file, output_dir):
    with tarfile.open(tar_file, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.startswith('uniref50.fasta'):
                member.name = os.path.basename(member.name)
                tar.extract(member, output_dir)
                return os.path.join(output_dir, member.name)
                
def process_fasta(input_file, output_file, lengths_and_offsets_file):
    offsets = []
    lengths = []
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Skip header
        for line in infile:
            if line.startswith('>'):
                break
        
        # Process sequences
        for line in tqdm(infile, desc="Processing FASTA"):
            if line.startswith('>'):
                offset = outfile.tell()
                offsets.append(offset)
                outfile.write(line)
            else:
                seq = line.strip()
                lengths.append(len(seq))
                outfile.write(seq + '\n')
    
    np.savez(lengths_and_offsets_file, seq_offsets=np.array(offsets), seq_lengths=np.array(lengths))
    
def create_splits(total_sequences, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    indices = list(range(total_sequences))
    random.shuffle(indices)
    
    train_size = int(total_sequences * train_ratio)
    valid_size = int(total_sequences * valid_ratio)
    
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size+valid_size]
    test_indices = indices[train_size+valid_size:]
    
    splits = {
        'train': train_indices,
        'valid': valid_indices,
        'test': test_indices
    }
    
    return splits

def main():
    # Create data directory
    data_dir = '/vast/aa11803/uniref50_data/'
    os.makedirs(data_dir, exist_ok=True)
    
    # Download UniRef50 2020_01 release
    url = 'https://ftp.ebi.ac.uk/pub/databases/uniprot/previous_releases/release-2020_01/uniref/uniref2020_01.tar.gz'
    filename = os.path.join(data_dir, 'uniref2020_01.tar.gz')
    download_file(url, filename)
    
    # Extract UniRef50 FASTA file from the tar.gz archive
    uniref50_fasta = extract_uniref50(filename, data_dir)
    # uniref50_fasta = os.path.join(data_dir, 'uniref50.tar')
    
    # Process FASTA file
    consensus_file = os.path.join(data_dir, 'consensus.fasta')
    lengths_and_offsets_file = os.path.join(data_dir, 'lengths_and_offsets.npz')
    process_fasta(uniref50_fasta, consensus_file, lengths_and_offsets_file)
    
    # Create splits
    metadata = np.load(lengths_and_offsets_file)
    total_sequences = len(metadata['seq_offsets'])
    splits = create_splits(total_sequences)
    
    # Save splits
    splits_file = os.path.join(data_dir, 'splits.json')
    with open(splits_file, 'w') as f:
        json.dump(splits, f)
        
    print(f"Processing complete. Data saved in {data_dir}")
    print(f"Total sequences: {total_sequences}")
    print(f"Train set size: {len(splits['train'])}")
    print(f"Validation set size: {len(splits['valid'])}")
    print(f"Test set size: {len(splits['test'])}")

if __name__ == '__main__':
    main()