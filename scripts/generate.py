# scripts/generate.py
#!/usr/bin/env python3
"""Text generation script for yacto-GPT."""

import os
import sys
import argparse
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import Generator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Generate text with yacto-GPT')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='To be or not to be',
        help='Input prompt'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/inference.yaml',
        help='Path to inference config'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='Top-k sampling'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p (nucleus) sampling'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Stream output token by token'
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create generator
    print(f"Loading model from {args.checkpoint}...")
    generator = Generator(
        model_path=args.checkpoint,
        device=config['inference'].get('device', 'auto'),
        dtype=config['inference'].get('dtype', 'float16')
    )
    
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    
    if args.stream:
        # Stream generation
        print("Generated:", end=" ")
        for token in generator.generate_stream(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        ):
            print(token, end="", flush=True)
        print()
    else:
        # Batch generation
        results = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_samples=args.num_samples
        )
        
        for i, result in enumerate(results):
            if args.num_samples > 1:
                print(f"\nSample {i+1}:")
                print("-" * 30)
            print(result)


if __name__ == '__main__':
    main()
