# scripts/prepare_data.py
#!/usr/bin/env python3
"""Prepare and preprocess datasets for yacto-GPT."""

import os
import sys
import argparse
import urllib.request

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_shakespeare(output_path: str = 'data/raw/shakespeare.txt'):
    """Download the Shakespeare dataset."""
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download file
    print(f"Downloading Shakespeare dataset from {url}...")
    urllib.request.urlretrieve(url, output_path)
    
    # Get file info
    file_size = os.path.getsize(output_path)
    with open(output_path, 'r', encoding='utf-8') as f:
        text = f.read()
        n_chars = len(text)
        n_unique = len(set(text))
        
    print(f"Downloaded to {output_path}")
    print(f"File size: {file_size:,} bytes")
    print(f"Total characters: {n_chars:,}")
    print(f"Unique characters: {n_unique}")
    print(f"First 100 characters: {text[:100]}...")


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for yacto-GPT')
    parser.add_argument(
        '--dataset',
        type=str,
        default='shakespeare',
        choices=['shakespeare'],
        help='Dataset to prepare'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/shakespeare.txt',
        help='Output path for dataset'
    )
    args = parser.parse_args()
    
    if args.dataset == 'shakespeare':
        download_shakespeare(args.output)
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)


if __name__ == '__main__':
    main()